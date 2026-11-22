"""Whole images through an upscaler: overlapping tiles, tiling mode, and alpha."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from arc_upscale.upscalers import Upscaler

DEFAULT_TILE = 256  # input pixels per tile side
TILE_OVERLAP = 16  # input pixels blended between neighbouring tiles
TILING_PAD = 16  # input pixels of wrapped border in tiling mode


def tile_starts(length: int, tile: int) -> list[int]:
    """Start offsets of tiles that cover `length`, neighbours sharing at least TILE_OVERLAP."""
    if length <= tile:
        return [0]
    return [*range(0, length - tile, tile - TILE_OVERLAP), length - tile]


def feather(length: int, ramp: int, at_start: bool, at_end: bool) -> torch.Tensor:
    """1-D blend weights: a linear ramp on each edge that touches a neighbouring tile."""
    weights = torch.ones(length)
    ramp = min(ramp, length // 2)
    if ramp > 0:
        slope = torch.linspace(0, 1, ramp + 2)[1:-1]
        if not at_start:
            weights[:ramp] = slope
        if not at_end:
            weights[-ramp:] = slope.flip(0)
    return weights


def upscale_tiled(upscaler: Upscaler, image: torch.Tensor, tile: int) -> torch.Tensor:
    """Upscale a (1, 3, H, W) tensor in overlapping tiles, blending the overlaps."""
    scale = upscaler.scale
    _, _, height, width = image.shape
    out = torch.zeros(1, 3, height * scale, width * scale)
    weight = torch.zeros(1, 1, height * scale, width * scale)
    for y in tile_starts(height, tile):
        for x in tile_starts(width, tile):
            patch = image[:, :, y : y + tile, x : x + tile]
            ph, pw = patch.shape[2:]
            wy = feather(ph * scale, TILE_OVERLAP * scale, y == 0, y + ph == height)
            wx = feather(pw * scale, TILE_OVERLAP * scale, x == 0, x + pw == width)
            mask = (wy[:, None] * wx[None, :])[None, None]
            region = (..., slice(y * scale, (y + ph) * scale), slice(x * scale, (x + pw) * scale))
            out[region] += upscaler(patch) * mask
            weight[region] += mask
    return (out / weight).clamp_(0, 1)


def fold_wrap(result: torch.Tensor, border: int, dim: int) -> torch.Tensor:
    """Cut the wrapped border off both ends of `dim`, blending each into the opposite edge.

    Each border is a second rendering of the band at the opposite edge. A model
    that invents detail renders the two differently, and cutting the border off
    would leave a seam in the detail. So the two are cross-faded, half and half
    at the very edge.
    """
    start, inner, end = result.split([border, result.shape[dim] - 2 * border, border], dim=dim)
    shape = [1, 1, 1, 1]
    shape[dim] = border
    keep = torch.linspace(0.5, 1, border).view(shape)  # weight of the inner rendering, from the edge inwards
    inner = inner.clone()
    first, last = inner.narrow(dim, 0, border), inner.narrow(dim, inner.shape[dim] - border, border)
    first.copy_(first * keep + end * (1 - keep))
    last.copy_(last * keep.flip(dim) + start * (1 - keep.flip(dim)))
    return inner


def upscale_tensor(upscaler: Upscaler, image: torch.Tensor, tile: int, tiling: bool) -> torch.Tensor:
    """Upscale a (1, 3, H, W) tensor. Tiling mode makes opposite edges match."""
    if not tiling:
        return upscale_tiled(upscaler, image, tile)
    pad = min(TILING_PAD, *image.shape[2:])
    wrapped = torch.nn.functional.pad(image, (pad, pad, pad, pad), mode="circular")
    # The tile grows by the border, so an image that fits one tile still fits one.
    result = upscale_tiled(upscaler, wrapped, tile + 2 * pad)
    border = pad * upscaler.scale
    return fold_wrap(fold_wrap(result, border, dim=2), border, dim=3)


def upscale_image(
    upscaler: Upscaler, image: Image.Image, tile: int = DEFAULT_TILE, tiling: bool = False
) -> Image.Image:
    """Upscale an 8-bit image. Transparency is kept."""
    has_alpha = image.has_transparency_data
    array = np.asarray(image.convert("RGBA" if has_alpha else "RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)[None]

    channels = [upscale_tensor(upscaler, tensor[:, :3], tile, tiling)]
    if has_alpha:
        alpha = tensor[:, 3:]
        if bool((alpha == 1).all()):
            channels.append(torch.ones(1, 1, *channels[0].shape[2:]))
        else:
            # Alpha goes through the model as a grey image. With the PBRify models, on
            # two masks, this was 2 to 7 dB closer to the truth than plain resampling.
            # It was not measured for SeedVR2.
            grey = upscale_tensor(upscaler, alpha.expand(-1, 3, -1, -1), tile, tiling)
            channels.append(grey.mean(dim=1, keepdim=True))
    merged = torch.cat(channels, dim=1)[0].permute(1, 2, 0)
    return Image.fromarray((merged * 255.0).round().to(torch.uint8).numpy())
