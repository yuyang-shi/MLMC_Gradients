import os

import contextlib
import urllib
from pathlib import Path
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision.datasets
import torchvision.transforms as transforms

import imageio

from .supervised_dataset import SupervisedDataset


# Returns tuple of form `(images, labels)`. Both are uint8 tensors. (Yuyang: modified test_dset to float tensor)
# `images` has shape `(nimages, nchannels, nrows, ncols)`, and has
# entries in {0, ..., 255}
def get_raw_image_tensors(dataset_name, train, data_root):
    data_dir = os.path.join(data_root, dataset_name)

    if dataset_name in ["mnist", "fashion-mnist"]:
        dataset_class = {
            "mnist": torchvision.datasets.MNIST,
            "fashion-mnist": torchvision.datasets.FashionMNIST
        }[dataset_name]
        dataset = dataset_class(root=data_dir, train=train, download=True)
        images = dataset.data.unsqueeze(1)
        labels = dataset.targets
        images = images.to(torch.float)

    elif dataset_name == "omniglot":
        import scipy.io
        try:
            raw_data = scipy.io.loadmat(data_root + "omniglot/chardata.mat")
        except FileNotFoundError:
            Path(data_root + "omniglot/").mkdir(parents=True, exist_ok=True)
            url = "https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat"
            print('Downloading from {}...'.format(url))
            local_filename = data_root + "omniglot/chardata.mat"
            urllib.request.urlretrieve(url, local_filename)
            print('Saved to {}'.format(local_filename))
            raw_data = scipy.io.loadmat(data_root + "omniglot/chardata.mat")

        if train:
            @contextlib.contextmanager
            def temp_seed(seed):
                state = np.random.get_state()
                np.random.seed(seed)
                try:
                    yield
                finally:
                    np.random.set_state(state)

            with temp_seed(0):
                perm = np.random.permutation(24345)
            print(perm)
            assert np.array_equal(perm[:3], np.array([9825, 22946, 16348]))
            images = np.transpose(raw_data["data"])[perm]
            images = torch.from_numpy(images)
            labels = np.transpose(raw_data["targetchar"])[perm]
            labels = torch.from_numpy(labels)
        else:
            images = np.transpose(raw_data["testdata"])
            images = torch.from_numpy(images)
            labels = np.transpose(raw_data["testtargetchar"])
            labels = torch.from_numpy(labels)

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return images.to(torch.float), labels.to(torch.uint8)


def image_tensors_to_supervised_dataset(dataset_name, dataset_role, images, labels):
    images = images.to(dtype=torch.get_default_dtype())
    labels = labels.long()
    return SupervisedDataset(dataset_name, dataset_role, images, labels)


def get_train_valid_image_datasets(dataset_name, data_root, valid_fraction, add_train_hflips):
    images, labels = get_raw_image_tensors(dataset_name, train=True, data_root=data_root)

    valid_size = int(valid_fraction * images.shape[0])
    valid_images = images[images.shape[0]-valid_size:]
    valid_labels = labels[images.shape[0]-valid_size:]
    train_images = images[:images.shape[0]-valid_size]
    train_labels = labels[:images.shape[0]-valid_size]

    if add_train_hflips:
        train_images = torch.cat((train_images, train_images.flip([3])))
        train_labels = torch.cat((train_labels, train_labels))

    train_dset = image_tensors_to_supervised_dataset(dataset_name, "train", train_images, train_labels)
    valid_dset = image_tensors_to_supervised_dataset(dataset_name, "valid", valid_images, valid_labels)

    return train_dset, valid_dset


def get_test_image_dataset(dataset_name, data_root):
    images, labels = get_raw_image_tensors(dataset_name, train=False, data_root=data_root)
    return image_tensors_to_supervised_dataset(dataset_name, "test", images, labels)


def get_image_datasets(dataset_name, data_root, **kwargs):
    if kwargs["make_valid_loader"]:
        if dataset_name in ("mnist", "fashion-mnist"):
            valid_fraction = 1 / 6
        elif dataset_name == "omniglot":
            valid_fraction = 1345 / 24345
        else:
            valid_fraction = 0.1
    else:
        valid_fraction = 0
    add_train_hflips = False

    train_dset, valid_dset = get_train_valid_image_datasets(dataset_name, data_root, valid_fraction, add_train_hflips)
    test_dset = get_test_image_dataset(dataset_name, data_root)

    return train_dset, valid_dset, test_dset
