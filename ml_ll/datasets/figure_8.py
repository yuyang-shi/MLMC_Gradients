from pathlib import Path
import numpy as np
import torch
from .supervised_dataset import SupervisedDataset


def get_figure_8_data(data, size):
    if data == "figure-8":
        z = torch.randn(size)
        u = np.pi * (0.6 + 1.8 * torch.distributions.Normal(0, 1).cdf(z))
        fz = torch.stack((np.sqrt(2) / 2 * torch.cos(u) / (torch.sin(u) ** 2 + 1),
                          np.sqrt(2) * torch.cos(u) * torch.sin(u) / (torch.sin(u) ** 2 + 1)), 1)
        data = fz + np.sqrt(0.02) * torch.randn_like(fz)

    else:
        assert False, f"Unknown dataset `{data}''"

    return torch.tensor(data, dtype=torch.get_default_dtype())


def get_figure_8_datasets(name, data_root, **kwargs):
    try:
        train_dset, valid_dset, test_dset = torch.load(data_root + "2d/" + name + ".pt")
        print("Loaded dataset " + name)
    except FileNotFoundError:
        Path(data_root + "2d/").mkdir(parents=True, exist_ok=True)
        train_dset = SupervisedDataset(name, "train", get_figure_8_data(name, size=5000))
        valid_dset = SupervisedDataset(name, "valid", get_figure_8_data(name, size=2000))
        test_dset = SupervisedDataset(name, "test", get_figure_8_data(name, size=2000))
        torch.save([train_dset, valid_dset, test_dset], data_root + "2d/" + name + ".pt")
        print("Generated new dataset " + name)
    return train_dset, valid_dset, test_dset
