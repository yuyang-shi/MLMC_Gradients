from .distributions import independent_bernoulli, independent_normal
from .model import LVM
from .networks import MLP, MLP_Normal, MLP_Normal_Constant_Sigma, Linear_Normal, Linear_Normal_Constant_Sigma, \
    Linear_Normal_Constant_Diagonal, Constant_Normal_Constant_Sigma, Identity_Normal_Constant_Sigma
import torch
from functools import partial


def get_network_and_distribution(schema):
    if schema["distribution_type"] == "Normal":
        if schema["network_type"] == "MLP_Normal":
            network = MLP_Normal(schema["net_dims"], get_nonlinearity(schema["nonlinearity"]))
        elif schema["network_type"] == "MLP_Normal_Constant_Sigma":
            network = MLP_Normal_Constant_Sigma(schema["net_dims"], get_nonlinearity(schema["nonlinearity"]),
                                                schema["init_logsigma"], schema["fix_logsigma"])
        elif schema["network_type"] == "Linear_Normal":
            network = Linear_Normal(schema["net_dims"])
        elif schema["network_type"] == "Linear_Normal_Constant_Sigma":
            network = Linear_Normal_Constant_Sigma(schema["net_dims"], schema["init_logsigma"], schema["fix_logsigma"])
        elif schema["network_type"] == "Linear_Normal_Constant_Diagonal":
            network = Linear_Normal_Constant_Diagonal(schema["net_dims"], schema["init_logdiagonal"],
                                                      schema["fix_logdiagonal"])
        elif schema["network_type"] == "Constant_Normal_Constant_Sigma":
            network = Constant_Normal_Constant_Sigma(schema["net_dims"], schema["init_mu"], schema["init_logsigma"],
                                                     schema["fix_mu"], schema["fix_logsigma"])
        elif schema["network_type"] == "Identity_Normal_Constant_Sigma":
            network = Identity_Normal_Constant_Sigma(schema["net_dims"], schema["init_logsigma"],
                                                     schema["fix_logsigma"])
        distribution = independent_normal

    elif schema["distribution_type"] == "Bernoulli":
        if schema["network_type"] == "MLP":
            network = MLP(schema["net_dims"], get_nonlinearity(schema["nonlinearity"]))
        # Takes in logit as distribution parameter
        distribution = independent_bernoulli

    return network, distribution


def get_model(schema):
    print(schema)
    decoder, px_z_dist_fn = get_network_and_distribution(schema["decoder"])
    prior, pz_dist_fn = get_network_and_distribution(schema["prior"])
    encoder, qz_x_dist_fn = get_network_and_distribution(schema["encoder"])
    return LVM(decoder, prior, encoder, px_z_dist_fn, pz_dist_fn, qz_x_dist_fn)


def get_nonlinearity(nonlinearity_name):
    if nonlinearity_name == "tanh":
        return torch.nn.Tanh()
    elif nonlinearity_name == "leaky_ReLU":
        return torch.nn.LeakyReLU(inplace=True)
    elif nonlinearity_name == "ReLU":
        return torch.nn.ReLU(True)
