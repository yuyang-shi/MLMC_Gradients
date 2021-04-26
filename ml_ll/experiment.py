import json
import random
from pathlib import Path
from functools import partial

import sys
import os
import subprocess

import numpy as np
import scipy.stats as stats
import pandas as pd
from matplotlib import pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn

from .trainer import Trainer
from .datasets import get_loaders
from .visualizer import DummyDensityVisualizer, MNISTDensityVisualizer, TwoDimensionalDensityVisualizer
from .models import get_model
from .writer import Writer, DummyWriter
from .metrics import *

from config import get_schema


def train(config, resume_dir):
    model, trainer, writer = setup_experiment(config=config, resume_dir=resume_dir)

    writer.write_json("config", config)

    writer.write_json("model", {
        "num_params": num_params(model),
        "schema": get_schema(config)
    })

    print("\nConfig:")
    print(json.dumps(config, indent=4))
    print(f"\nNumber of parameters: {num_params(model):,}\n")

    trainer.train()


def print_test_metrics(config, resume_dir, visualize=False):
    _, trainer, _ = setup_experiment(
        config={**config, "write_to_disk": False},
        resume_dir=resume_dir,
        load_latest=config["load_latest"]
    )

    with torch.no_grad():
        test_metrics = trainer.test(visualize)

    test_metrics = {k: v.item() for k, v in test_metrics.items()}

    print(json.dumps(test_metrics, indent=4))


def print_test_metrics_verbose(config, resume_dir):
    model, trainer, writer = setup_experiment(
        config={**config, "write_to_disk": resume_dir is None},
        resume_dir=resume_dir,
        load_latest=config["load_latest"]
    )

    test_metrics = trainer.test_verbose(config["decay_rate_test_num_samples"], config["decay_rate_test_K_max"],
                                        config["decay_rate_test_grad_idx"],
                                        config["tvo_integrand_shape_test_num_partitions"],
                                        "sumo" in config["train_objective"],
                                        "tvo" in config["train_objective"])

    test_metrics = {k: v.item() for k, v in test_metrics.items()}

    print(json.dumps(test_metrics, indent=4))

    K_max = config["decay_rate_test_K_max"]

    nll_delta_decay_np = np.array([test_metrics["nll-Delta-" + str(k) + "-squared"] for k in range(K_max + 1)])
    nll_delta_batch_avg_decay_np = np.array(
        [test_metrics["nll-Delta-" + str(k) + "-squared-batch-averaged"] for k in range(K_max + 1)])
    p_grad_delta_batch_avg_decay_np = np.array(
        [test_metrics["p-grad-Delta-" + str(k) + "-squared-batch-averaged"] for k in range(K_max + 1)])

    delta_decay_list = [nll_delta_decay_np, nll_delta_batch_avg_decay_np, p_grad_delta_batch_avg_decay_np]
    name_list = ["NLL Decay Rate", "NLL Decay Rate (Batch Averaged)", "p Grad Decay Rate (Batch Averaged)"]

    decay_rate_raw_results_dict = {}
    for delta_decay_np, name in zip(delta_decay_list, name_list):
        delta_decay_rate_exponent = stats.linregress(np.arange(K_max + 1), np.log2(delta_decay_np))[0]

        plt.figure()
        plt.semilogy(np.arange(K_max + 1), delta_decay_np,
                     label=r'$E[\Delta_k^2] \approx O(2^{{ {:.2f} k}})$, '
                           '\n'
                           r'$r_{{ max }} \approx {:.2f}$'.format(
                         delta_decay_rate_exponent, 1 - 2 ** delta_decay_rate_exponent))
        plt.semilogy(np.arange(K_max + 1), delta_decay_np[0] / 2 ** np.arange(K_max + 1), label=r'$O(2^{-k})$',
                     alpha=0.5, color="red")
        plt.semilogy(np.arange(K_max + 1), delta_decay_np[0] / (2 ** 1.5) ** np.arange(K_max + 1),
                     label=r'$O(2^{-1.5k})$', alpha=0.5, color="orange")
        plt.semilogy(np.arange(K_max + 1), delta_decay_np[0] / 4 ** np.arange(K_max + 1), label=r'$O(2^{-2k})$',
                     alpha=0.5, color="green")

        plt.xlabel("$k$")
        plt.title(name)
        plt.legend()
        plt.savefig(os.path.join(trainer._writer._logdir, name + ".pdf"), format="pdf")

        decay_rate_raw_results_dict[name] = delta_decay_np

    torch.save(decay_rate_raw_results_dict, os.path.join(trainer._writer._logdir, "decay_rate_raw_results_dict.pt"))

    if resume_dir is None:
        trainer._save_checkpoint("latest")
        writer.write_json("config", config)

        writer.write_json("model", {
            "num_params": num_params(model),
            "schema": get_schema(config)
        })


def perform_bias_var_test(config, resume_dir):
    model, trainer, _ = setup_experiment(
        config={**config, "write_to_disk": False},
        resume_dir=resume_dir,
        load_latest=config["load_latest"]
    )
    x_data = trainer._train_loader.dataset.x

    print(f"\nNumber of parameters: {num_params(model):,}\n")

    test_quantity_list = config["bias_var_test_quantity_list"]
    if config["dataset"] == "tom-linear-gaussian":
        results_df_name_list = (["snr_log", "bias_log", "var_log"] if "nll" in test_quantity_list else []) + \
                               (["snr_p", "bias_p", "var_p"] if "p_grad" in test_quantity_list else []) + \
                               (["snr_q", "bias_q", "var_q"] if "q_grad" in test_quantity_list else [])
    else:
        results_df_name_list = (["snr_log", "var_log"] if "nll" in test_quantity_list else []) + \
                               (["snr_p", "var_p"] if "p_grad" in test_quantity_list else []) + \
                               (["snr_q", "var_q"] if "q_grad" in test_quantity_list else [])

    results_df_dict = {}

    raw_results_list = []

    for df_name in results_df_name_list:
        results_df_dict[df_name] = pd.DataFrame(columns=[config["train_objective_comment"]],
                                                index=range(config["num_bias_var_test_experiments"]))

    if config["dataset"] == "tom-linear-gaussian":
        SAVE_PATH = "results/" + config["dataset"] + "/dim" + str(config["latent_dim"]) + "_sig" + str(
            config["model_perturb_sig"]) + "/"
    else:
        SAVE_PATH = "results/" + config["dataset"] + "/dim" + str(config["latent_dim"]) + "/"
    Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

    mean_sampling_cost = int(round(compute_ml_ll_exp_term(config)))

    for i in range(config["num_bias_var_test_experiments"]):
        model, trainer, _ = setup_experiment(
            config={**config, "write_to_disk": False},
            resume_dir=resume_dir,
            load_latest=config["load_latest"]
        )

        if config["dataset"] == "tom-linear-gaussian":
            torch.manual_seed(i)

            model.prior.mu.data = torch.mean(x_data, dim=0)
            model.encoder.linear_module_mu.weight.data = torch.eye(config["latent_dim"]).to(trainer._device) / 2
            model.encoder.linear_module_mu.bias.data = torch.mean(x_data, dim=0) / 2

            model.prior.mu.data = model.prior.mu.data + torch.randn_like(model.prior.mu.data) * config[
                "model_perturb_sig"]
            model.encoder.linear_module_mu.weight.data = model.encoder.linear_module_mu.weight.data + torch.randn_like(
                model.encoder.linear_module_mu.weight.data) * config["model_perturb_sig"]
            model.encoder.linear_module_mu.bias.data = model.encoder.linear_module_mu.bias.data + torch.randn_like(
                model.encoder.linear_module_mu.bias.data) * config["model_perturb_sig"]

            torch.random.seed()

            true_nll = - model.compute_true_logprob(x_data)
            print("True NLL: " + str(true_nll.item()))

            true_nll.backward()
            true_p_grad_vector = model.get_p_grad()
            true_p_grad_entry = true_p_grad_vector[i].item()
            print("True p_param logprob grad: " + str(true_p_grad_entry))
            model.zero_grad()

            true_kl_p_q = model.compute_true_kl_p_q(x_data)
            true_kl_p_q.backward()
            true_kl_q_grad_vector = model.get_q_grad()
            true_kl_q_grad_entry = true_kl_q_grad_vector[i].item()
            print("True q_param KL grad: " + str(true_kl_q_grad_entry))
            model.zero_grad()

        bias_var_test_results = trainer.bias_var_test(config["num_bias_var_test_grad_samples"], test_quantity_list)

        if "nll" in test_quantity_list:
            nll_results = bias_var_test_results["nll"]
            nll_results_mean = nll_results.mean().item()
            results_df_dict["snr_log"].iloc[i] = nll_results_mean / nll_results.std().item()
            results_df_dict["var_log"].iloc[i] = nll_results.var().item()
            if config["dataset"] == "tom-linear-gaussian":
                results_df_dict["bias_log"].iloc[i] = (nll_results_mean - true_nll.item()) ** 2

        # Use the i-th entry for gradient result
        if "p_grad" in test_quantity_list:
            p_grad_results = bias_var_test_results["p_grad"]
            p_grad_entry_results = p_grad_results[:, i]
            p_grad_results_mean = p_grad_entry_results.mean().item()
            results_df_dict["snr_p"].iloc[i] = p_grad_results_mean / p_grad_entry_results.std().item()
            results_df_dict["var_p"].iloc[i] = p_grad_entry_results.var().item()
            if config["dataset"] == "tom-linear-gaussian":
                results_df_dict["bias_p"].iloc[i] = (p_grad_results_mean - true_p_grad_entry) ** 2

        if "q_grad" in test_quantity_list:
            q_grad_results = bias_var_test_results["q_grad"]
            q_grad_entry_results = q_grad_results[:, i]
            q_grad_results_mean = q_grad_entry_results.mean().item()
            results_df_dict["snr_q"].iloc[i] = q_grad_results_mean / q_grad_entry_results.std().item()
            results_df_dict["var_q"].iloc[i] = q_grad_entry_results.var().item()
            if config["dataset"] == "tom-linear-gaussian":
                results_df_dict["bias_q"].iloc[i] = (q_grad_results_mean - true_kl_q_grad_entry) ** 2

        if config["dataset"] == "tom-linear-gaussian":
            raw_results_list.append({
                "true_nll": true_nll.item(),
                "true_p_grad": true_p_grad_vector,
                "true_q_grad": true_kl_q_grad_vector,
                **bias_var_test_results
            })
        else:
            raw_results_list.append(bias_var_test_results)

    try:
        for df_name, df in results_df_dict.items():
            df1 = pd.read_csv(SAVE_PATH + df_name + str(mean_sampling_cost) + ".csv")
            cols = [col for col in df1.columns if col not in df.columns]
            pd.concat([df1[cols], df], axis=1).to_csv(
                SAVE_PATH + df_name + str(mean_sampling_cost) + ".csv", index=False)
    except FileNotFoundError:
        for df_name, df in results_df_dict.items():
            df.to_csv(SAVE_PATH + df_name + str(mean_sampling_cost) + ".csv", index=False)

    try:
        raw_results_dict = torch.load(SAVE_PATH + "raw_results_dict" + str(mean_sampling_cost) + ".pt")
    except FileNotFoundError:
        raw_results_dict = {}

    raw_results_dict[config["train_objective_comment"]] = raw_results_list
    torch.save(raw_results_dict, SAVE_PATH + "raw_results_dict" + str(mean_sampling_cost) + ".pt")


def print_model(config):
    model, _, _, _ = setup_model_and_loaders(
        config={**config, "write_to_disk": False},
        device=torch.device("cpu")
    )
    print(model)


def print_num_params(config):
    model, _, _, _ = setup_model_and_loaders(
        config={**config, "write_to_disk": False},
        device=torch.device("cpu")
    )
    print(f"Number of parameters: {num_params(model):,}")


def setup_model_and_loaders(config, device):
    train_loader, valid_loader, test_loader = get_loaders(
        dataset_name=config["dataset"],
        device=device,
        data_root=config["data_root"],
        make_valid_loader=config["early_stopping"],
        train_batch_size=config["train_batch_size"],
        valid_batch_size=config["valid_batch_size"],
        test_batch_size=config["test_batch_size"],
        latent_dim=config["latent_dim"],
        obs_dim=config["obs_dim"]
    )
    print(train_loader.dataset.x.shape)
    print(valid_loader.dataset.x.shape)
    print(test_loader.dataset.x.shape)
    model = get_model(
        schema=get_schema(config=config, ),
    )

    model.to(device)
    print(model)
    x_data = train_loader.dataset.x

    if config["dataset"] == "tom-linear-gaussian":
        with torch.random.fork_rng():
            torch.manual_seed(0)

            model.prior.mu.data = torch.mean(x_data, dim=0)
            model.encoder.linear_module_mu.weight.data = torch.eye(config["latent_dim"]).to(device) / 2
            model.encoder.linear_module_mu.bias.data = torch.mean(x_data, dim=0) / 2

            model.prior.mu.data = model.prior.mu.data + torch.randn_like(model.prior.mu.data) * config[
                "model_perturb_sig"]
            model.encoder.linear_module_mu.weight.data = model.encoder.linear_module_mu.weight.data + torch.randn_like(
                model.encoder.linear_module_mu.weight.data) * config["model_perturb_sig"]
            model.encoder.linear_module_mu.bias.data = model.encoder.linear_module_mu.bias.data + torch.randn_like(
                model.encoder.linear_module_mu.bias.data) * config["model_perturb_sig"]

    return model, train_loader, valid_loader, test_loader


def setup_experiment(config, resume_dir, load_latest=True,
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"] + 1)
    random.seed(config["seed"] + 2)

    model, train_loader, valid_loader, test_loader = setup_model_and_loaders(
        config=config,
        device=device
    )

    if resume_dir is None:
        if config["dataset"] in ("mnist", "omniglot", "fashion-mnist") and \
                type(model.decoder).__name__ == "MLP":
            print("Heuristically set the final layer of decoder network. ")
            if config["dataset"] in ("mnist", "fashion-mnist"):
                x_data = train_loader.dataset.x / 255
            else:
                x_data = train_loader.dataset.x
            mean_img = x_data.mean(0).view(-1).cpu()
            mean_img = np.clip(mean_img, 1e-8, 1.0 - 1e-7)
            mean_img_logit = np.log(mean_img / (1.0 - mean_img))
            model.decoder.fc_modules[-1].bias.data = torch.tensor(mean_img_logit, device=device)

    if config["opt"] == "sgd":
        opt_class = optim.SGD
    elif config["opt"] == "rmsprop":
        opt_class = optim.RMSprop
    elif config["opt"] == "adam":
        opt_class = optim.Adam
    elif config["opt"] == "amsgrad":
        opt_class = partial(optim.Adam, eps=1e-4, amsgrad=True)
    else:
        assert False, f"Invalid optimiser type {config['opt']}"

    if config["write_to_disk"]:
        if resume_dir is None:
            logdir = config["logdir_root"]
            make_subdir = True
        else:
            logdir = resume_dir
            make_subdir = False

        writer = Writer(logdir=logdir, make_subdir=make_subdir, tag_group=config["dataset"])
    else:
        writer = DummyWriter(logdir=resume_dir)

    if config["dataset"] in ["mnist", "fashion-mnist", "omniglot"]:
        visualizer = MNISTDensityVisualizer(writer=writer)
    elif train_loader.dataset.x.shape[1:] == (2,):
        try:
            visualizer = TwoDimensionalDensityVisualizer(
                writer=writer,
                x_train=train_loader.dataset.x,
                num_elbo_samples=config["num_test_elbo_samples"],
                device=device,
                visualization_method=config["2d_density_visualization_method"]
            )
        except KeyError:
            visualizer = TwoDimensionalDensityVisualizer(
                writer=writer,
                x_train=train_loader.dataset.x,
                num_elbo_samples=config["num_test_elbo_samples"],
                device=device
            )
    else:
        visualizer = DummyDensityVisualizer(writer=writer)

    train_metrics = get_train_metrics(config)

    def valid_metrics(density, x):
        return partial(metrics, density=density, x=x, num_elbo_samples=config["num_valid_elbo_samples"])

    def test_metrics(density, x):
        return partial(metrics, density=density, x=x, num_elbo_samples=config["num_test_elbo_samples"])

    opt = opt_class(
        [{"params": model.p_params, "lr": config["lr_p"], "weight_decay": config["weight_decay_p"]},
         {"params": model.q_params, "lr": config["lr_q"], "weight_decay": config["weight_decay_q"]}]
    )

    if config["lr_schedule"] == "linear":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=opt,
            lr_lambda=lambda epoch: max(1 - config["lr_schedule_decay_factor"] * epoch / 3280, 0.1),
            verbose=False
        )
    elif config["lr_schedule"] == "none":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=opt,
            lr_lambda=lambda epoch: 1.
        )
    else:
        assert False, f"Invalid learning rate schedule `{config['lr_schedule']}'"

    trainer = Trainer(
        model=model,
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        opt=opt,
        lr_scheduler=lr_scheduler,
        max_epochs=config["max_epochs"],
        obs_dim=config["obs_dim"],
        max_grad_norm_p=config["max_grad_norm_p"],
        max_grad_norm_q=config["max_grad_norm_q"],
        early_stopping=config["early_stopping"],
        max_bad_valid_epochs=config["max_bad_valid_epochs"],
        visualizer=visualizer,
        writer=writer,
        epochs_per_valid=config["epochs_per_valid"],
        epochs_per_test=config["epochs_per_test"],
        should_checkpoint_latest=config["should_checkpoint_latest"],
        should_checkpoint_best_valid=config["should_checkpoint_best_valid"],
        load_latest=load_latest,
        device=device,
        dataset=config["dataset"]
    )

    return model, trainer, writer


def compute_ml_ll_exp_term(config):
    if config["train_objective"] in ["iwae", "elbo", "tvo"]:
        return config["num_importance_samples"]
    elif config["train_objective"] == "sumo":
        return config["sumo_min_importance_samples"] + 5
    _K_dist = stats.geom(p=config["ml_ll_geom_prob"], loc=-1)

    if config["ml_ll_K_cap"] == np.inf:
        # Checked
        _mean_sampling_cost = 2 ** (config["ml_ll_I0_start_level"] + 1) * config["ml_ll_geom_prob"] / (
                1 - 2 * (1 - config["ml_ll_geom_prob"]))
    else:
        _mean_sampling_cost = np.sum(
            2 ** (np.arange(config["ml_ll_K_cap"] + 1) + config["ml_ll_I0_start_level"] + 1) * _K_dist.pmf(
                np.arange(config["ml_ll_K_cap"] + 1)) / _K_dist.cdf(config["ml_ll_K_cap"]))

    print(f"\nTarget: IWAE_{2 ** (config['ml_ll_K_cap'] + config['ml_ll_I0_start_level'] + 1):,}")
    print(f"Expected sampling cost: {_mean_sampling_cost:,}")
    return _mean_sampling_cost


def get_train_metrics(config):
    if config["train_objective"] == "iwae":
        train_metrics = IWAE(config["num_importance_samples"], config["q_train_method"])

    elif config["train_objective"] == "elbo":
        train_metrics = ELBO(config["num_importance_samples"], config["q_train_method"])

    elif config["train_objective"] in ("ml-ll-ss", "ml-ll-rr"):
        compute_ml_ll_exp_term(config)
        train_metrics = ML_LL(config["train_objective"].split("-")[2], config["ml_ll_geom_prob"], config["ml_ll_K_cap"],
                              config["ml_ll_I0_start_level"], config["q_train_method"],
                              config["q_num_importance_samples"])

    elif config["train_objective"] == "sumo":
        train_metrics = SUMO(config["sumo_min_importance_samples"], config["q_train_method"],
                             config["q_num_importance_samples"])

    elif config["train_objective"] == "tvo":
        train_metrics = TVO(config["num_importance_samples"], config["q_train_method"], config["num_partitions"],
                            config["partition_type"], config["log_beta_min"])

    elif config["train_objective"] in ("ml-tvo-ss", "ml-tvo-rr"):
        compute_ml_ll_exp_term(config)
        train_metrics = ML_TVO(config["train_objective"].split("-")[2], config["ml_ll_geom_prob"],
                               config["ml_ll_K_cap"],
                               config["ml_ll_I0_start_level"], config["p_train_method"],
                               config["q_train_method"], config["q_num_importance_samples"],
                               config["num_partitions"], config["partition_type"], config["log_beta_min"])

    else:
        assert False, f"Invalid training objective `{config['train_objective']}'"

    return train_metrics


def num_params(module):
    return sum(p.view(-1).shape[0] for p in module.parameters())

