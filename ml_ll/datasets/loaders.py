import sys

import torch

from .figure_8 import get_figure_8_datasets
from .image import get_image_datasets
from .linear_gaussian import get_linear_gaussian_datasets


def get_loader(dset, device, batch_size, drop_last):
    return torch.utils.data.DataLoader(
        dset.to(device),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=False
    )


def get_loaders(
        dataset_name,
        device,
        data_root,
        make_valid_loader,
        train_batch_size,
        valid_batch_size,
        test_batch_size,
        **kwargs
):
    print("Loading data...", end="", flush=True, file=sys.stderr)

    if dataset_name in ["mnist", "fashion-mnist", "omniglot"]:
        train_dset, valid_dset, test_dset = get_image_datasets(dataset_name, data_root, make_valid_loader=make_valid_loader, **kwargs)

    elif dataset_name in ["tom-linear-gaussian"]:
        train_dset, valid_dset, test_dset = get_linear_gaussian_datasets(dataset_name, data_root, **kwargs)

    else:
        train_dset, valid_dset, test_dset = get_figure_8_datasets(dataset_name, data_root, **kwargs)

    print("Done.", file=sys.stderr)

    train_loader = get_loader(train_dset, device, train_batch_size, drop_last=True)

    if make_valid_loader:
        valid_loader = get_loader(valid_dset, device, valid_batch_size, drop_last=False)
    else:
        valid_loader = None

    test_loader = get_loader(test_dset, device, test_batch_size, drop_last=False)

    return train_loader, valid_loader, test_loader
