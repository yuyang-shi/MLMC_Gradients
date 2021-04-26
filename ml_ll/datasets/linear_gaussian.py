from pathlib import Path

import torch
from torch.distributions import MultivariateNormal
from .supervised_dataset import SupervisedDataset
import numpy as np


def get_linear_gaussian_data(name, size, latent_dim, obs_dim):
    if name == "tom-linear-gaussian":
        assert latent_dim == obs_dim
        theta_true = torch.ones(latent_dim)

        z = MultivariateNormal(theta_true, torch.eye(latent_dim)).sample((size,))  # (num_data, dim)
        x = MultivariateNormal(z, torch.eye(latent_dim)).sample()  # (num_data, dim)
    return x


def get_linear_gaussian_datasets(name, data_root, **kwargs):
    try:
        train_dset, valid_dset, test_dset = torch.load(
            data_root + "linear_gaussian/" + name + f"{kwargs['latent_dim']}_{kwargs['obs_dim']}.pt")
        print("Loaded dataset " + name + f"{kwargs['latent_dim']}_{kwargs['obs_dim']}.pt")
    except FileNotFoundError:
        Path(data_root + "linear_gaussian/").mkdir(parents=True, exist_ok=True)
        train_dset = SupervisedDataset(name, "train", get_linear_gaussian_data(name, 1024, kwargs['latent_dim'], kwargs['obs_dim']))
        valid_dset = SupervisedDataset(name, "valid", get_linear_gaussian_data(name, 1024, kwargs['latent_dim'], kwargs['obs_dim']))
        test_dset = SupervisedDataset(name, "test", get_linear_gaussian_data(name, 1024, kwargs['latent_dim'], kwargs['obs_dim']))
        torch.save([train_dset, valid_dset, test_dset],
                   data_root + "linear_gaussian/" + name + f"{kwargs['latent_dim']}_{kwargs['obs_dim']}.pt")
        print("Generated new dataset " + name + f"{kwargs['latent_dim']}_{kwargs['obs_dim']}.pt")
    return train_dset, valid_dset, test_dset
