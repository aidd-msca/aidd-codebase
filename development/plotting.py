from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

# from coding_framework_MVP.utils.typescripts import Tensor
from matplotlib.colors import ListedColormap


class _ABSPlotter:
    def __init__(self, filepath: str) -> None:
        self.fig, self.axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        self.n_axes = len(self.axes)

        self.filepath = filepath

    def set_labels(self, xlabel: str, ylabel: str, plot_idx: int = 0) -> None:
        self.axes[plot_idx].set_xlabel(xlabel)
        self.axes[plot_idx].set_ylabel(ylabel)

    def set_title(self, title: str, plot_idx: int = 0, **kwargs) -> None:
        self.axes[plot_idx].set_title(title, **kwargs)

    def set_figure_title() -> None:
        pass

    def set_axis(self, plot_idx: int = 0, **kwargs) -> None:
        self.axes[plot_idx].axis(**kwargs)

    def set_axis_equal(self, plot_idx: int = 0) -> None:
        self.axes[plot_idx].axis("equal")

    def add_diagonal(self, plot_idx: int = 0) -> None:
        self.axes[plot_idx].plot(
            [0, 1], [0, 1], transform=self.axes[plot_idx].transAxes
        )

    def save_figure(self) -> None:
        self.fig.savefig(self.filepath)

    def scatterplot(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        plot_idx: int = 0,
        **kwargs
    ) -> None:
        max_range = max(max(actual), max(predicted))
        min_range = min(min(actual), min(predicted))
        self.axes[plot_idx].scatter(x=actual, y=predicted, **kwargs)

        plt.plot([max_range, min_range])

        plt.imshow(
            heatmap_array,
            cmap=my_cmap,
            vmin=-max_min_range,
            vmax=max_min_range,
            interpolation="nearest",
        )
        plt.savefig(filepath, bbox_inches="tight")


def heatmap(
    heatmap_array: np.ndarray,
    filepath: str,
    autosize: bool = True,
    w_px: int = 1024,
    h_px: int = 1024,
):
    max_min_range = np.abs(heatmap_array).max()

    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap = ListedColormap(my_cmap)

    if autosize:
        plt.figure()
    else:
        px = 1 / plt.rcParams["figure.dpi"]
        plt.figure(figsize=(w_px * px, h_px * px))
        # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.axis("off")
    plt.imshow(
        heatmap_array,
        cmap=my_cmap,
        vmin=-max_min_range,
        vmax=max_min_range,
        interpolation="nearest",
    )
    plt.savefig(filepath, bbox_inches="tight")
