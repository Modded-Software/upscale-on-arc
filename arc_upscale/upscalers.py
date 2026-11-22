"""Models loaded onto one Intel GPU, each callable on a single patch."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Protocol

import torch
from spandrel import ImageModelDescriptor, ModelLoader

from arc_upscale.catalog import SEEDVR2_SEED, SEEDVR2_VAE_OVERLAP, SEEDVR2_VAE_TILE, SEEDVR2_WEIGHTS
from arc_upscale.errors import UserError
from arc_upscale.fetch import fetch_seedvr2


class Upscaler(Protocol):
    """Maps a (1, 3, H, W) CPU tensor in [0, 1] to (1, 3, H*scale, W*scale)."""

    scale: int

    def __call__(self, patch: torch.Tensor) -> torch.Tensor: ...


class XpuUpscaler:
    """A model file loaded onto one Intel GPU."""

    def __init__(self, model_file: Path, device: torch.device) -> None:
        try:
            descriptor = ModelLoader().load_from_file(model_file)
        except Exception as error:  # each file format fails with its own exception type
            raise UserError(f"cannot load model file {model_file}: {error}") from error
        if not isinstance(descriptor, ImageModelDescriptor):
            raise UserError(f"{model_file} is not a single-image upscaling model")
        self._descriptor = descriptor.to(device).eval()
        self._device = device
        self.scale = descriptor.scale

    @torch.inference_mode()
    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        return self._descriptor(patch.to(self._device)).cpu()


class SeedVR2Upscaler:
    """SeedVR2 on one Intel GPU. It drives upstream's four phases, as upstream's own CLI does."""

    scale = 4

    def __init__(self, device: torch.device, directory: Path | None = None) -> None:
        _code, weights = fetch_seedvr2(directory)
        with contextlib.redirect_stdout(sys.stderr):  # upstream logs with print()
            from src.core import generation_phases, generation_utils
            from src.utils.debug import Debug

            self._phases, self._utils, self._debug = generation_phases, generation_utils, Debug(enabled=False)
            # Upstream moves a model to its offload device after use. Naming the GPU
            # itself there keeps both models loaded between calls.
            self._devices = dict.fromkeys(
                ("dit_device", "vae_device", "dit_offload_device", "vae_offload_device"), device
            )
            context = generation_utils.setup_generation_context(**self._devices, debug=self._debug)
            self._runner, self._cache = generation_utils.prepare_runner(
                dit_model=SEEDVR2_WEIGHTS[0],
                vae_model=SEEDVR2_WEIGHTS[1],
                model_dir=str(weights),
                debug=self._debug,
                ctx=context,
                dit_cache=True,
                vae_cache=True,
                dit_id=1,
                vae_id=1,
                encode_tiled=True,
                encode_tile_size=SEEDVR2_VAE_TILE,
                encode_tile_overlap=SEEDVR2_VAE_OVERLAP,
                decode_tiled=True,
                decode_tile_size=SEEDVR2_VAE_TILE,
                decode_tile_overlap=SEEDVR2_VAE_OVERLAP,
            )

    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        phases, debug = self._phases, self._debug
        frames = patch[0].permute(1, 2, 0)[None].contiguous()  # (1, H, W, 3): a video of one frame
        # Upstream sizes its output by the short side.
        size = {"resolution": min(patch.shape[2:]) * self.scale, "max_resolution": 0}
        batch = {"batch_size": 1, "seed": SEEDVR2_SEED}
        with contextlib.redirect_stdout(sys.stderr):
            context = self._utils.setup_generation_context(**self._devices, debug=debug)
            context["cache_context"] = self._cache
            frames, _info = self._utils.compute_generation_info(
                ctx=context, images=frames, debug=debug, **batch, **size
            )
            # Upstream frees the allocator's cache after each phase, but for CUDA and MPS
            # only. On XPU the cache would grow past the card's memory. That gave
            # corrupt output without an error, or "out of host memory" on a later patch.
            context = phases.encode_all_batches(
                self._runner, ctx=context, images=frames, debug=debug, color_correction="lab", **batch, **size
            )
            torch.xpu.empty_cache()
            context = phases.upscale_all_batches(
                self._runner, ctx=context, debug=debug, seed=SEEDVR2_SEED, cache_model=True
            )
            torch.xpu.empty_cache()
            context = phases.decode_all_batches(self._runner, ctx=context, debug=debug, cache_model=True)
            torch.xpu.empty_cache()
            context = phases.postprocess_all_batches(ctx=context, debug=debug, color_correction="lab", batch_size=1)
        result = context["final_video"].to("cpu", torch.float32)  # (1, H*4, W*4, 3)
        del context
        torch.xpu.empty_cache()
        if not bool(torch.isfinite(result).all()):
            raise UserError(
                "SeedVR2 returned invalid values. Here that happened when GPU memory ran short: use a smaller --tile"
            )
        return result.permute(0, 3, 1, 2).clamp_(0, 1)
