"""Perlin noise main functionality."""
from math import pi
from typing import Callable, Tuple

import torch
from torch import Generator, Tensor, lerp
from torch.nn.functional import unfold


def get_positions(block_shape: Tuple[int, int]) -> Tensor:
    """
    Generate position tensor.

    Arguments:
        block_shape -- (height, width) of position tensor

    Returns:
        position vector shaped (1, height, width, 1, 1, 2)
    """
    bh, bw = block_shape
    positions = torch.stack(
        torch.meshgrid(
            [(torch.arange(b) + 0.5) / b for b in (bw, bh)],
            indexing="xy",
        ),
        -1,
    ).view(1, bh, bw, 1, 1, 2)
    return positions


def unfold_grid(vectors: Tensor) -> Tensor:
    """
    Unfold vector grid to batched vectors.

    Arguments:
        vectors -- grid vectors

    Returns:
        batched grid vectors
    """
    batch_size, _, gpy, gpx = vectors.shape
    return (
        unfold(vectors, (2, 2))
        .view(batch_size, 2, 4, -1)
        .permute(0, 2, 3, 1)
        .view(batch_size, 4, gpy - 1, gpx - 1, 2)
    )


def smooth_step(t: Tensor) -> Tensor:
    """
    Smooth step function [0, 1] -> [0, 1].

    Arguments:
        t -- input values (any shape)

    Returns:
        output values (same shape as input values)
    """
    return t * t * (3.0 - 2.0 * t)


def perlin_noise_tensor(
    vectors: Tensor, positions: Tensor, step: Callable = None
) -> Tensor:
    """
    Generate perlin noise from batched vectors and positions.

    Arguments:
        vectors -- batched grid vectors shaped (batch_size, 4, grid_height, grid_width, 2)
        positions -- batched grid positions shaped (batch_size or 1, block_height, block_width, grid_height or 1, grid_width or 1, 2)

    Keyword Arguments:
        step -- smooth step function [0, 1] -> [0, 1] (default: `smooth_step`)

    Raises:
        Exception: if position and vector shapes do not match

    Returns:
        (batch_size, block_height * grid_height, block_width * grid_width)
    """
    if step is None:
        step = smooth_step

    batch_size = vectors.shape[0]
    # grid height, grid width
    gh, gw = vectors.shape[2:4]
    # block height, block width
    bh, bw = positions.shape[1:3]

    for i in range(2):
        if positions.shape[i + 3] not in (1, vectors.shape[i + 2]):
            raise Exception(
                f"Blocks shapes do not match: vectors ({vectors.shape[1]}, {vectors.shape[2]}), positions {gh}, {gw})"
            )

    if positions.shape[0] not in (1, batch_size):
        raise Exception(
            f"Batch sizes do not match: vectors ({vectors.shape[0]}), positions ({positions.shape[0]})"
        )

    vectors = vectors.view(batch_size, 4, 1, gh * gw, 2)
    positions = positions.view(positions.shape[0], bh * bw, -1, 2)

    step_x = step(positions[..., 0])
    step_y = step(positions[..., 1])

    row0 = lerp(
        (vectors[:, 0] * positions).sum(dim=-1),
        (vectors[:, 1] * (positions - positions.new_tensor((1, 0)))).sum(dim=-1),
        step_x,
    )
    row1 = lerp(
        (vectors[:, 2] * (positions - positions.new_tensor((0, 1)))).sum(dim=-1),
        (vectors[:, 3] * (positions - positions.new_tensor((1, 1)))).sum(dim=-1),
        step_x,
    )
    noise = lerp(row0, row1, step_y)
    return (
        noise.view(
            batch_size,
            bh,
            bw,
            gh,
            gw,
        )
        .permute(0, 3, 1, 4, 2)
        .reshape(batch_size, gh * bh, gw * bw)
    )


def perlin_noise(
    grid_shape: Tuple[int, int],
    out_shape: Tuple[int, int],
    batch_size: int = 1,
    generator: Generator = None,
    *args,
    **kwargs,
) -> Tensor:
    """
    Generate perlin noise with given shape. `*args` and `**kwargs` are forwarded to `Tensor` creation.

    Arguments:
        grid_shape -- Shape of grid (height, width).
        out_shape -- Shape of output noise image (height, width).

    Keyword Arguments:
        batch_size -- (default: {1})
        generator -- random generator used for grid vectors (default: {None})

    Raises:
        Exception: if grid and out shapes do not match

    Returns:
        Noise image shaped (batch_size, height, width)
    """
    # grid height and width
    gh, gw = grid_shape
    # output height and width
    oh, ow = out_shape
    # block height and width
    bh, bw = oh // gh, ow // gw

    if oh != bh * gh:
        raise Exception(f"Output height {oh} must be divisible by grid height {gh}")
    if ow != bw * gw != 0:
        raise Exception(f"Output width {ow} must be divisible by grid width {gw}")

    angle = torch.empty(
        [batch_size] + [s + 1 for s in grid_shape], *args, **kwargs
    ).uniform_(to=2.0 * pi, generator=generator)
    # random vectors on grid points
    vectors = unfold_grid(torch.stack((torch.cos(angle), torch.sin(angle)), dim=1))
    # positions inside grid cells [0, 1)
    positions = get_positions((bh, bw)).to(vectors)
    return perlin_noise_tensor(vectors, positions).squeeze(0)
