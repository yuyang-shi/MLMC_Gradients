from .dsl import group, base, provides
import numpy as np

group(
    "tom-linear-gaussian",
    [
        "tom-linear-gaussian"
    ]
)


@base
def config(dataset):
    return {
        "latent_dim": 20,
        "obs_dim": [20],

        "max_epochs": 10,
        "max_grad_norm": None,
        "max_grad_norm_p": None,
        "max_grad_norm_q": None,
        "early_stopping": True,
        "max_bad_valid_epochs": 250,
        "train_batch_size": 256,
        "valid_batch_size": 256,
        "test_batch_size": 256,

        "opt": "adam",
        "lr_p": 1e-3,
        "lr_q": 1e-3,
        "lr_schedule": "none",
        "weight_decay_p": 0.,
        "weight_decay_q": 0.,
        "epochs_per_valid": 1,
        "epochs_per_test": 5,

        "num_valid_elbo_samples": 5000,
        "num_test_elbo_samples": 5000,

        "model_perturb_sig": 0.01,

        "decay_rate_test_num_samples": 100,
        "decay_rate_test_K_max": 10,
        "decay_rate_test_grad_idx": 0,

        "tvo_integrand_shape_test_num_partitions": 50,

        "num_bias_var_test_experiments": 10,
        "num_bias_var_test_grad_samples": 1000,
        "bias_var_test_quantity_list": ["nll", "p_grad", "q_grad"],

        "decoder":
            {"network_type": "Identity_Normal_Constant_Sigma",
             "net_dims": [],
             "distribution_type": "Normal",
             "init_logsigma": 0.,
             "fix_logsigma": True},
        "prior":
            {"network_type": "Constant_Normal_Constant_Sigma",
             "net_dims": [],
             "distribution_type": "Normal",
             "init_mu": 1.,
             "init_logsigma": 0.,
             "fix_mu": False,
             "fix_logsigma": True},
        "encoder":
            {"network_type": "Linear_Normal_Constant_Sigma",
             "net_dims": [],
             "distribution_type": "Normal",
             "init_logsigma": np.log(2 / 3) / 2.,
             "fix_logsigma": True},
    }


@provides("iwae")
def iwae(dataset, model):
    return {
        "train_objective": "iwae",
        "train_objective_comment": "iwae",
        "num_importance_samples": 6,
        "q_train_method": "iwae",
    }


@provides("rws")
def iwae(dataset, model):
    return {
        "train_objective": "iwae",
        "train_objective_comment": "rws",
        "num_importance_samples": 6,
        "q_train_method": "rws",
    }


@provides("elbo")
def elbo(dataset, model):
    return {
        "train_objective": "elbo",
        "num_importance_samples": 6,
        "q_train_method": "elbo",
    }


@provides("ml-ll-ss")
def ml_ll_ss(dataset, model):
    return {
        "train_objective": "ml-ll-ss",
        "train_objective_comment": "ml-ll-ss",
        "ml_ll_geom_prob": 0.6,
        "ml_ll_K_cap": np.inf,
        "ml_ll_I0_start_level": 0,
        "q_train_method": "unbiased rws",
        "q_num_importance_samples": None,
        "control_multiplier": 0,
    }


@provides("ml-ll-rr")
def ml_ll_rr(dataset, model):
    return {
        "train_objective": "ml-ll-rr",
        "train_objective_comment": "ml-ll-rr",
        "ml_ll_geom_prob": 0.6,
        "ml_ll_K_cap": np.inf,
        "ml_ll_I0_start_level": 0,
        "q_train_method": "unbiased rws",
        "q_num_importance_samples": None,
        "control_multiplier": 0,
    }


@provides("sumo")
def sumo(dataset, model):
    return {
        "train_objective": "sumo",
        "train_objective_comment": "sumo",
        "sumo_min_importance_samples": 1,
        "q_train_method": "unbiased rws",
        "q_num_importance_samples": None,
    }
