import os
from contextlib import suppress
from collections import Counter
from itertools import chain
import sys

import numpy as np

import torch
import torch.nn.utils

from ignite.engine import Events, Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import RunningAverage, Metric, Loss, Average
from ignite.handlers import TerminateOnNan
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, GradsScalarHandler


def preprocess_data(x, dataset_name, obs_dim):
    if dataset_name in ("mnist", "fashion-mnist"):
        x /= 255.
        x = torch.bernoulli(x)
    if dataset_name == "omniglot":
        x = torch.bernoulli(x)
    x = x.view((-1, *obs_dim))
    return x


class AverageMetric(Metric):
    # XXX: This is not ideal, since we are overriding a protected attribute in Metric.
    # However, as of ignite v0.3.0, this is necessary to allow us to return a
    # map from the Engines we attach this to. (In particular, note that e.g.
    # `Trainer._train_batch` should return a map of the form `{"metrics": METRICS_MAP}`.)
    _required_output_keys = ["metrics"]

    def reset(self):
        """ By default, this is called at the start of each epoch. """
        self._sums = Counter()
        self._num_examples = Counter()

    def update(self, output):
        """ By default, this is called once for each batch. """
        metrics, = output
        for k, v in metrics.items():
            self._sums[k] += torch.sum(v)
            self._num_examples[k] += torch.numel(v)

    def compute(self):
        return {k: v / self._num_examples[k] for k, v in self._sums.items()}

    def completed(self, engine):
        """ Helper method to compute metric's value and put into the engine. """
        engine.state.metrics = {**engine.state.metrics, **self.compute()}

    def attach(self, engine):
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)  # Triggers reset
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed)  # Update engine.state.metrics computation every iteration


class Trainer:
    _STEPS_PER_WRITE = 100

    def __init__(
            self,

            model,
            device,

            train_metrics,
            train_loader,
            opt,
            lr_scheduler,
            max_epochs,
            obs_dim,
            max_grad_norm_p,
            max_grad_norm_q,

            test_metrics,
            test_loader,
            epochs_per_valid,
            epochs_per_test,

            early_stopping,
            valid_metrics,
            valid_loader,
            max_bad_valid_epochs,

            visualizer,

            writer,
            should_checkpoint_latest,
            should_checkpoint_best_valid,
            load_latest,

            dataset
    ):
        self._model = model

        self._device = device

        self._train_metrics = train_metrics
        self._train_loader = train_loader
        self._opt = opt
        self._lr_scheduler = lr_scheduler
        self._max_epochs = max_epochs
        self._obs_dim = obs_dim
        self._max_grad_norm_p = max_grad_norm_p
        self._max_grad_norm_q = max_grad_norm_q

        self.raw_grad_norm_p = None
        self.raw_grad_norm_q = None

        self._test_metrics = test_metrics
        self._test_loader = test_loader
        self._epochs_per_valid = epochs_per_valid
        self._epochs_per_test = epochs_per_test

        self._valid_metrics = valid_metrics
        self._valid_loader = valid_loader
        self._max_bad_valid_epochs = max_bad_valid_epochs
        self._best_valid_loss = float("inf")
        self._num_bad_valid_epochs = 0

        self._visualizer = visualizer

        self._writer = writer
        self._should_checkpoint_best_valid = should_checkpoint_best_valid

        self._dataset = dataset

        ### Training

        self._trainer = Engine(self._train_batch)

        AverageMetric().attach(self._trainer)

        ProgressBar(persist=True).attach(self._trainer, ["p"])

        self._trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        self._trainer.add_event_handler(Events.ITERATION_COMPLETED, self._log_training_info)

        ### Validation

        if early_stopping:
            self._validator = Engine(self._validate_batch)

            AverageMetric().attach(self._validator)
            ProgressBar(persist=False, desc="Validating").attach(self._validator)

            self._trainer.add_event_handler(Events.EPOCH_COMPLETED, self._validate)

        ### Testing

        self._tester = Engine(self._test_batch)
        AverageMetric().attach(self._tester)
        ProgressBar(persist=False, desc="Testing").attach(self._tester)

        self._trainer.add_event_handler(Events.EPOCH_COMPLETED, self._test_and_log)

        ### Checkpointing

        if should_checkpoint_latest:
            self._trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: self._save_checkpoint("latest"))

        # Load checkpoints at the last step
        if load_latest:
            try:
                self._load_checkpoint("latest")
            except FileNotFoundError:
                print("Did not find `latest' checkpoint.", file=sys.stderr)
        else:
            try:
                self._load_checkpoint("best_valid")
            except FileNotFoundError:
                print("Did not find `best_valid' checkpoint.", file=sys.stderr)

    # Main training loop
    def train(self):
        self._trainer.run(data=self._train_loader, max_epochs=self._max_epochs)

    def _train_batch(self, engine, batch):
        self._model.train()

        x, _ = batch
        x = x.to(self._device)
        x = preprocess_data(x, self._dataset, self._obs_dim)
        self._opt.zero_grad()

        all_values = self._train_metrics(self._model, x)

        losses = all_values["losses"]

        if "metrics" in all_values:
            metrics = all_values["metrics"]

            shared_keys = set(losses.keys()).intersection(metrics.keys())
            assert not shared_keys, f"Shared metrics and losses keys: {shared_keys}"

        else:
            metrics = {}

        loss_p = losses["p"]
        loss_q = losses["q"]

        p_grad = torch.autograd.grad(loss_p, self._model.p_params, retain_graph=True)
        q_grad = torch.autograd.grad(loss_q, self._model.q_params)
        torch.autograd.backward(self._model.p_params, p_grad)
        torch.autograd.backward(self._model.q_params, q_grad)

        if self._max_grad_norm_p is not None:
            self.raw_grad_norm_p = torch.nn.utils.clip_grad_norm_(self._model.p_params, self._max_grad_norm_p)
        if self._max_grad_norm_q is not None:
            self.raw_grad_norm_q = torch.nn.utils.clip_grad_norm_(self._model.q_params, self._max_grad_norm_q)

        self._opt.step()

        return {"metrics": {**metrics, **losses}}


    def test(self, visualize=False, num_test_runs=1):
        if visualize:
            with torch.no_grad():
                self._visualizer.visualize(self._model, -1, testing=True)
        return self._tester.run(data=chain.from_iterable([self._test_loader]*num_test_runs)).metrics


    def test_verbose(self, decay_rate_test_num_samples, decay_rate_test_K_max, decay_rate_test_grad_idx,
                     tvo_integrand_shape_test_num_partitions=None,
                     if_sumo=False, if_tvo=False, num_test_runs=1):
        self.decay_rate_test_num_samples = decay_rate_test_num_samples
        self.decay_rate_test_K_max = decay_rate_test_K_max
        self.decay_rate_test_grad_idx = decay_rate_test_grad_idx
        self.tvo_integrand_shape_test_num_partitions = tvo_integrand_shape_test_num_partitions
        self.if_sumo = if_sumo
        self.if_tvo = if_tvo

        self._tester_verbose = Engine(self._test_batch_verbose)
        AverageMetric().attach(self._tester_verbose)
        ProgressBar(persist=False, desc="Testing").attach(self._tester_verbose)

        self._model.train()
        with torch.no_grad():
            self._visualizer.visualize(self._model, -1, testing=True)
        return self._tester_verbose.run(data=chain.from_iterable([self._test_loader]*num_test_runs)).metrics

    def bias_var_test(self, num_grad_samples, test_quantity_list):
        # Bias variance test.
        self.bias_var_test_results = {}

        self._bias_var_tester = Engine(self._bias_var_test_batch)

        for test_quantity in test_quantity_list:
            assert test_quantity in ["nll", "p_grad", "q_grad"]
            if test_quantity == "nll":
                Average(output_transform=lambda output: output["nll"]).attach(self._bias_var_tester, "nll")
            elif test_quantity == "p_grad":
                Average(output_transform=lambda output: output["p_grad"]).attach(self._bias_var_tester, "p_grad")
            elif test_quantity == "q_grad":
                Average(output_transform=lambda output: output["q_grad"]).attach(self._bias_var_tester, "q_grad")

        self._bias_var_tester.add_event_handler(Events.EPOCH_COMPLETED, self._aggregate_bias_var_test_results)
        self._bias_var_tester.add_event_handler(Events.EPOCH_STARTED, self._train_metrics_update_K)

        ProgressBar(persist=False, desc="Testing").attach(self._bias_var_tester)

        # Main loop
        self._bias_var_tester.run(data=self._train_loader, max_epochs=num_grad_samples)
        return self.bias_var_test_results

    def _test_and_log(self, engine):
        epoch = engine.state.epoch
        if (epoch - 1) % self._epochs_per_test == 0:  # Test after first epoch
            for k, v in self.test().items():
                self._writer.write_scalar(f"test/{k}", v, global_step=engine.state.epoch)

                if not torch.isfinite(v):
                    self._save_checkpoint(tag="nan_during_test")
            with torch.no_grad():
                self._visualizer.visualize(self._model, epoch)

    def _test_batch(self, engine, batch):
        x, _ = batch
        x = x.to(self._device)
        x = preprocess_data(x, self._dataset, self._obs_dim)
        return {"metrics": self._test_metrics(self._model, x)(train_metrics=self._train_metrics)}

    def _test_batch_verbose(self, engine, batch):
        x, _ = batch
        x = x.to(self._device)
        x = preprocess_data(x, self._dataset, self._obs_dim)
        return {"metrics": self._test_metrics(self._model, x)(verbose=True, train_metrics=self._train_metrics,
                                                              decay_rate_test_num_samples=self.decay_rate_test_num_samples,
                                                              decay_rate_test_K_max=self.decay_rate_test_K_max,
                                                              decay_rate_test_grad_idx=self.decay_rate_test_grad_idx,
                                                              tvo_integrand_shape_test_num_partitions=self.tvo_integrand_shape_test_num_partitions,
                                                              if_sumo=self.if_sumo,
                                                              if_tvo=self.if_tvo
                                                              )}

    def _bias_var_test_batch(self, engine, batch):
        self._model.train()

        x, _ = batch
        x = x.to(self._device)
        x = preprocess_data(x, self._dataset, self._obs_dim)

        self._model.zero_grad()

        losses = self._train_metrics(self._model, x, update_K=False)["losses"]

        loss_p = losses["p"]
        loss_q = losses["q"]

        p_grad = torch.autograd.grad(loss_p, self._model.p_params, retain_graph=True)
        q_grad = torch.autograd.grad(loss_q, self._model.q_params)
        torch.autograd.backward(self._model.p_params, p_grad)
        torch.autograd.backward(self._model.q_params, q_grad)

        p_grad = self._model.get_p_grad()
        q_grad = self._model.get_q_grad()

        return {"nll": loss_p.view(1, 1), "p_grad": p_grad, "q_grad": q_grad}

    def _aggregate_bias_var_test_results(self, engine):
        # Triggered on EPOCH_COMPLETED
        for k, v in engine.state.metrics.items():
            if k not in self.bias_var_test_results:
                self.bias_var_test_results[k] = engine.state.metrics[k].view(1, -1)
            else:
                self.bias_var_test_results[k] = torch.cat([self.bias_var_test_results[k],
                                                           engine.state.metrics[k].view(1, -1)], dim=0)

    def _train_metrics_update_K(self, engine):
        # Triggered on EPOCH_STARTED
        self._train_metrics._update_K()


    def _validate(self, engine):
        # Triggered on EPOCH_COMPLETED
        lr_p, lr_q = self._get_lr()
        self._writer.write_scalar("train/lr_p", lr_p, global_step=engine.state.epoch)
        self._writer.write_scalar("train/lr_q", lr_q, global_step=engine.state.epoch)

        if (engine.state.epoch - 1) % self._epochs_per_valid == 0 or engine.state.epoch == self._max_epochs:  # Test after first epoch

            state = self._validator.run(data=self._valid_loader)
            valid_loss = state.metrics["loss"]

            self._lr_scheduler.step()

            self._writer.write_scalar("valid/loss", valid_loss, global_step=engine.state.epoch)

            if valid_loss < self._best_valid_loss:
                print(f"Best validation loss {valid_loss} after epoch {engine.state.epoch}")
                self._num_bad_valid_epochs = 0
                self._best_valid_loss = valid_loss

                if self._should_checkpoint_best_valid:
                    self._save_checkpoint(tag="best_valid")

            else:
                if not torch.isfinite(valid_loss):
                    self._save_checkpoint(tag="nan_during_validation")

                self._num_bad_valid_epochs += 1

                # We do this manually (i.e. don't use Ignite's early stopping) to permit
                # saving/resuming more easily
                if self._num_bad_valid_epochs > self._max_bad_valid_epochs:
                    print(
                        f"No validation improvement after {self._num_bad_valid_epochs} epochs. Terminating."
                    )
                    print(f"Final validation loss {valid_loss} after epoch {engine.state.epoch}")
                    self._trainer.terminate()
            if engine.state.epoch == self._max_epochs:
                print(f"Final validation loss {valid_loss} after epoch {engine.state.epoch}")

        else:
            self._lr_scheduler.step()


    def _validate_batch(self, engine, batch):
        x, _ = batch
        x = x.to(self._device)
        x = preprocess_data(x, self._dataset, self._obs_dim)
        all_values = self._valid_metrics(self._model, x)(train_metrics=self._train_metrics)
        return {"metrics": {"loss": - all_values["log-prob"]}}

    def _log_training_info(self, engine):
        i = engine.state.iteration

        if i % self._STEPS_PER_WRITE == 0:
            for k, v in engine.state.output["metrics"].items():
                self._writer.write_scalar("train/" + k, v, global_step=i)

            norm, norm_p, norm_q = self._get_grad_norms()

            self._writer.write_scalar("train/grad-norm_p", norm_p, global_step=i)
            self._writer.write_scalar("train/grad-norm_q", norm_q, global_step=i)

            if self.raw_grad_norm_p is not None:
                self._writer.write_scalar("train/raw-grad-norm_p", self.raw_grad_norm_p, global_step=i)
            if self.raw_grad_norm_q is not None:
                self._writer.write_scalar("train/raw-grad-norm_q", self.raw_grad_norm_q, global_step=i)

    def _get_grad_norms(self):
        norm_p = 0
        for param in self._model.p_params:
            if param.grad is not None:
                norm_p += param.grad.norm().item() ** 2
        norm_q = 0
        for param in self._model.q_params:
            if param.grad is not None:
                norm_q += param.grad.norm().item() ** 2
        norm = norm_p + norm_q
        return np.sqrt(norm), np.sqrt(norm_p), np.sqrt(norm_q)

    def _get_lr(self):
        param_group_p, param_group_q = self._opt.param_groups
        return param_group_p["lr"], param_group_q["lr"]

    def _save_checkpoint(self, tag):
        # We do this manually (i.e. don't use Ignite's checkpointing) because
        # Ignite only allows saving objects, not scalars (e.g. the current epoch)
        checkpoint = {
            "epoch": self._trainer.state.epoch,
            "iteration": self._trainer.state.iteration,
            "module_state_dict": self._model.state_dict(),
            "opt_state_dict": self._opt.state_dict(),
            "best_valid_loss": self._best_valid_loss,
            "num_bad_valid_epochs": self._num_bad_valid_epochs,
            "lr_scheduler_state_dict": self._lr_scheduler.state_dict()
        }

        self._writer.write_checkpoint(tag, checkpoint)

    def _load_checkpoint(self, tag):
        checkpoint = self._writer.load_checkpoint(tag, device=self._device)

        @self._trainer.on(Events.STARTED)
        def resume_trainer_state(engine):
            engine.state.epoch = checkpoint["epoch"]
            engine.state.iteration = checkpoint["iteration"]

        self._model.load_state_dict(checkpoint["module_state_dict"], strict=False)
        self._opt.load_state_dict(checkpoint["opt_state_dict"])
        self._best_valid_loss = checkpoint["best_valid_loss"]
        self._num_bad_valid_epochs = checkpoint["num_bad_valid_epochs"]
        try:
            self._lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        except KeyError:
            print("No lr scheduler in saved checkpoint")

        print(f"Loaded checkpoint `{tag}' after epoch {checkpoint['epoch']}", file=sys.stderr)
