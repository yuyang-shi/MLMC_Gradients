from collections import defaultdict

import numpy as np

import torch
import torch.utils.data
import torchvision.utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import tqdm

from .metrics import metrics


class DensityVisualizer:
    def __init__(self, writer):
        self._writer = writer

    def visualize(self, density, epoch, testing=False):
        raise NotImplementedError


class DummyDensityVisualizer(DensityVisualizer):
    def visualize(self, density, epoch, testing=False):
        return


class MNISTDensityVisualizer(DensityVisualizer):
    def visualize(self, density, epoch, testing=False):
        imgs = density.fixed_sample()
        imgs = imgs.view([-1, 1, 28, 28])
        num_images = imgs.shape[0]
        num_rows = int(np.ceil(num_images / min(np.sqrt(num_images), 10)))

        grid = torchvision.utils.make_grid(
            imgs, nrow=num_rows, pad_value=1,
            normalize=True, scale_each=True
        )
        if testing:
            def show(img):
                npimg = img.cpu().numpy()
                plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='binary', interpolation='nearest')
            show(grid)
        else:
            self._writer.write_image("samples", grid, global_step=epoch)


class TwoDimensionalDensityVisualizer(DensityVisualizer):
    _GRID_SIZE = 150
    _CONTOUR_LEVELS = 50
    _NUM_TRAIN_POINTS_TO_SHOW = 500
    _PADDING = .2
    _BATCH_SIZE = 1000

    def __init__(self, writer, x_train, num_elbo_samples, device, visualization_method="contourf"):
        super().__init__(writer=writer)

        self._x = x_train

        self._x1_lims = self._lims(self._x[:, 0])
        self._x2_lims = self._lims(self._x[:, 1])

        self._num_elbo_samples = num_elbo_samples

        self._device = device
        self.visualization_method = visualization_method

    def _lims(self, t):
        return (
            t.min().item() - self._PADDING,
            t.max().item() + self._PADDING
        )

    def visualize(self, density, epoch, testing=False):
        if self.visualization_method == "pcolor":
            _x_lims = (min(self._x1_lims[0], self._x2_lims[0]), max(self._x1_lims[1], self._x2_lims[1]))
            grid_x1, grid_x2 = torch.meshgrid((
                torch.linspace(*_x_lims, self._GRID_SIZE),
                torch.linspace(*_x_lims, self._GRID_SIZE)
            ))
        elif self.visualization_method == "contourf":
            grid_x1, grid_x2 = torch.meshgrid((
                torch.linspace(*self._x1_lims, self._GRID_SIZE),
                torch.linspace(*self._x2_lims, self._GRID_SIZE)
            ))
        else:
            assert False, f"Invalid visualizer visualization method `{self.visualization_method}'"
        x1_x2 = torch.stack((grid_x1, grid_x2), dim=2).view(-1, 2)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x1_x2.to(self._device)),
            batch_size=self._BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False
        )

        probs = []
        for x1_x2_batch, in tqdm.tqdm(loader, leave=False, desc="Plotting"):
            with torch.no_grad():
                log_prob = metrics(density, x1_x2_batch, self._num_elbo_samples)["log-prob"]
            probs.append(torch.exp(log_prob))

        probs = torch.cat(probs, dim=0).view(*grid_x1.shape).cpu()

        if self.visualization_method == "pcolor":
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.pcolormesh(grid_x1, grid_x2, probs, vmin=0, vmax=0.8, cmap="coolwarm")

        elif self.visualization_method == "contourf":
            contours = plt.contourf(grid_x1, grid_x2, probs, levels=self._CONTOUR_LEVELS, cmap="coolwarm")

            for c in contours.collections:
                c.set_edgecolor("face")
            cb = plt.colorbar()
            cb.solids.set_edgecolor("face")

        x = self._x.cpu()
        x = x[torch.randint(high=x.shape[0], size=(self._NUM_TRAIN_POINTS_TO_SHOW,))]
        plt.scatter(x[:, 0], x[:, 1], c="k", marker=".", s=7, linewidth=0.5, alpha=0.5)
        if testing:
            plt.show()
        else:
            self._writer.write_figure("density", plt.gcf(), epoch)

        plt.close()
