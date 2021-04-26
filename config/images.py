from .dsl import group, base, provides
import numpy as np

group(
    "images",
    [
        "mnist",
        "omniglot",
        "fashion-mnist"
    ]
)


@base
def config(dataset):
    return {
        "latent_dim": 50,
        "obs_dim": [784],

        "max_epochs": 3280,
        "max_grad_norm": None,
        "max_grad_norm_p": 60,
        "max_grad_norm_q": 125,
        "early_stopping": True,
        "max_bad_valid_epochs": 3280,
        "train_batch_size": 100,
        "valid_batch_size": 50,
        "test_batch_size": 50,

        "opt": "amsgrad",
        "lr_p": 1e-3,
        "lr_q": 1e-3,
        "lr_schedule": "linear",
        "lr_schedule_decay_factor": 0.9,
        "weight_decay_p": 0.,
        "weight_decay_q": 0.,
        "epochs_per_valid": 50,
        "epochs_per_test": 50,

        "num_valid_elbo_samples": 5000,
        "num_test_elbo_samples": 5000,

        "decay_rate_test_num_samples": 1,
        "decay_rate_test_K_max": 10,
        "decay_rate_test_grad_idx": 0,

        "tvo_integrand_shape_test_num_partitions": 50,

        "bias_var_test_quantity_list": ["nll", "p_grad"],
        "num_bias_var_test_experiments": 1,
        "num_bias_var_test_grad_samples": 10,

        "encoder":
            {"network_type": "MLP_Normal",
             "net_dims": [200, 200],
             "distribution_type": "Normal",
             "nonlinearity": "tanh"},
        "decoder":
            {"network_type": "MLP",
             "net_dims": [200, 200],
             "distribution_type": "Bernoulli",
             "nonlinearity": "tanh"}
    }


@provides("iwae5")
def iwae(dataset, model):
    return {
        "train_objective": "iwae",
        "num_importance_samples": 5,
        "q_train_method": "iwae",
    }


@provides("iwae15")
def iwae(dataset, model):
    return {
        "train_objective": "iwae",
        "num_importance_samples": 15,
        "q_train_method": "iwae",
    }


@provides("ml-ll-rr5")
def ml_ll(dataset, model):
    return {
        "train_objective": "ml-ll-rr",
        "ml_ll_geom_prob": 0.585,
        "ml_ll_K_cap": 6,
        "ml_ll_I0_start_level": 0,
        "q_train_method": "iwae",
        "q_num_importance_samples": 5,
        "control_multiplier": 0,
    }


@provides("ml-ll-rr15")
def ml_ll(dataset, model):
    return {
        "train_objective": "ml-ll-rr",
        "ml_ll_geom_prob": 0.63,
        "ml_ll_K_cap": 4,
        "ml_ll_I0_start_level": 2,
        "q_train_method": "iwae",
        "q_num_importance_samples": 15,
        "control_multiplier": 0,
    }


@provides("sumo5")
def sumo(dataset, model):
    return {
        "train_objective": "sumo",
        "sumo_min_importance_samples": 1,
        "q_train_method": "iwae",
        "q_num_importance_samples": 5,
    }


@provides("sumo15")
def sumo(dataset, model):
    return {
        "train_objective": "sumo",
        "sumo_min_importance_samples": 9,
        "q_train_method": "iwae",
        "q_num_importance_samples": 15,
    }


@provides("tvo")
def tvo(dataset, model):
    return {
        "train_objective": "tvo",
        "num_importance_samples": 5,
        "q_train_method": "iwae",
        "num_partitions": 5,
        "partition_type": "log",
        "log_beta_min": np.log10(0.03)
    }


@provides("ml-tvo-rr5")
def ml_tvo(dataset, model):
    return {
        "train_objective": "ml-tvo-rr",
        "ml_ll_geom_prob": 0.585,
        "ml_ll_K_cap": 6,
        "ml_ll_I0_start_level": 0,
        "p_train_method": "covariance",
        "q_train_method": "iwae",
        "q_num_importance_samples": 5,
        "num_partitions": 5,
        "partition_type": "log",
        "log_beta_min": np.log10(0.03)
    }


@provides("ml-tvo-rr15")
def ml_tvo(dataset, model):
    return {
        "train_objective": "ml-tvo-rr",
        "ml_ll_geom_prob": 0.63,
        "ml_ll_K_cap": 4,
        "ml_ll_I0_start_level": 2,
        "p_train_method": "covariance",
        "q_train_method": "iwae",
        "q_num_importance_samples": 15,
        "num_partitions": 5,
        "partition_type": "log",
        "log_beta_min": np.log10(0.01)
    }