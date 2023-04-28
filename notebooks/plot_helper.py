"""Helper functions for plotting in notebooks."""
import torch
from matplotlib import pyplot as plt
from torch import Tensor


def grid_plot(grid: Tensor, noise: Tensor) -> None:
    """
    Plot grid and vectors.

    Arguments:
        grid -- grid vectors
        noise -- noise image
    """
    h, w = noise.shape
    gh, gw = grid.shape[2:4]
    lx, ly = [torch.linspace(0, d, s) for s, d in zip((gw, gh), (w, h))]
    x, y = torch.meshgrid(
        [lx, ly],
        indexing="xy",
    )
    grid = grid.cpu()
    u = grid[0, 0]
    v = grid[0, 1]

    ax = plt.gca()
    ax.grid(color="black", alpha=0.5)
    ax.set_xticks(lx)
    ax.set_yticks(ly)
    ax.quiver(x, y, u, v, width=0.004, color="black")
    return ax.imshow(noise.cpu(), extent=[0, w, h, 0])
