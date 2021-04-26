from .dsl import group, base, provides, GridParams
import numpy as np

group(
    "figure-8",
    [
        "figure-8"
    ]
)


@base
def config(dataset):
    return {
        "latent_dim": 1,
        "obs_dim": [2],

        "max_epochs": 2000,
        "max_grad_norm": None,
        "max_grad_norm_p": None,
        "max_grad_norm_q": None,
        "early_stopping": True,
        "max_bad_valid_epochs": 2000,
        "train_batch_size": 100,
        "valid_batch_size": 1000,
        "test_batch_size": 1000,

        "opt": "adam",
        "lr_p": 2.5e-4,
        "lr_q": 2.5e-4,
        "lr_schedule": "none",
        "weight_decay_p": 0.,
        "weight_decay_q": 0.,
        "epochs_per_valid": 1,
        "epochs_per_test": 25,

        "num_valid_elbo_samples": 5000,
        "num_test_elbo_samples": 5000,

        "encoder":
            {"network_type": "MLP_Normal",
             "net_dims": [50, 50, 50],
             "distribution_type": "Normal",
             "nonlinearity": "leaky_ReLU"
             },
        "decoder":
            {"network_type": "MLP_Normal_Constant_Sigma",
             "net_dims": [50, 50, 50],
             "distribution_type": "Normal",
             "nonlinearity": "leaky_ReLU",
             "init_logsigma": np.log(0.02) / 2,
             "fix_logsigma": False},

        "decay_rate_test_num_samples": 1,
        "decay_rate_test_K_max": 10,
        "decay_rate_test_grad_idx": 0,

        "tvo_integrand_shape_test_num_partitions": 50,

        "2d_density_visualization_method": "pcolor"
    }


@provides("iwae")
def iwae(dataset, model):
    return {
        "train_objective": "iwae",
        "num_importance_samples": 5,
        "q_train_method": "iwae",
    }


@provides("ml-ll-ss")
def ml_ll_ss(dataset, model):
    return {
        "train_objective": "ml-ll-ss",
        "ml_ll_geom_prob": 0.625,
        "ml_ll_K_cap": np.inf,
        "ml_ll_I0_start_level": 0,
        "q_train_method": "var",
        "q_num_importance_samples": 5,
        "control_multiplier": 0,
    }


@provides("ml-ll-rr")
def ml_ll_rr(dataset, model):
    return {
        "train_objective": "ml-ll-rr",
        "ml_ll_geom_prob": 0.625,
        "ml_ll_K_cap": np.inf,
        "ml_ll_I0_start_level": 0,
        "q_train_method": "var",
        "q_num_importance_samples": 5,
        "control_multiplier": 0,
    }


@provides("sumo")
def sumo(dataset, model):
    return {
        "train_objective": "sumo",
        "sumo_min_importance_samples": 1,
        "q_train_method": "var",
        "q_num_importance_samples": 5,
    }