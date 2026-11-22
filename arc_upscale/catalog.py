"""The models arc-upscale knows by name, with the pinned files and checksums."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from arc_upscale.errors import UserError

# The project directory. The arc-upscale launcher and `make test` set it.
ROOT_VARIABLE = "ARC_UPSCALE_ROOT"
RELEASES = "https://github.com/Kim2091/Kim2091-Models/releases/download"


def models_dir() -> Path:
    """`models/` in the project named by ARC_UPSCALE_ROOT. Models and the SeedVR2 code are kept there."""
    root = os.environ.get(ROOT_VARIABLE, "")
    if not root:
        raise UserError(f"{ROOT_VARIABLE} is not set; run through the arc-upscale command, or make test")
    return Path(root) / "models"


class ModelFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    file: str
    url: str
    sha256: str
    summary: str


# Models that spandrel loads from one file.
# Upstream publishes no checksums. These pin the files this project was tested with.
MODELS = {
    "pbrify": ModelFile(
        file="4x-PBRify_UpscalerV4.safetensors",
        url=f"{RELEASES}/4x-PBRify_UpscalerV4/4x-PBRify_UpscalerV4.safetensors",
        sha256="a00fa50116895a9b766aeb012edad0a14d66f3afda65539efc0f6fc28c991ba2",
        summary="4x-PBRify_UpscalerV4 (DAT2, CC0), stays close to the input, fast",
    ),
    "fast": ModelFile(
        file="4x-PBRify_RPLKSRd_V3.pth",
        url=f"{RELEASES}/4x-PBRify_RPLKSRd_V3/4x-PBRify_RPLKSRd_V3.pth",
        sha256="1325a08842ebfc18aea90e71594aa9b9e82b9c7e16321dccf5b757884a22daf1",
        summary="4x-PBRify_RPLKSRd_V3 (RealPLKSR, CC0), fastest, softer",
    ),
}

# SeedVR2 is not a single file. It is upstream's code at a pinned commit, plus two
# weight files that upstream's own downloader fetches and checks against its SHA-256 list.
SEEDVR2 = "seedvr2"
SEEDVR2_REPO = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler"
SEEDVR2_COMMIT = "4490bd1f482e026674543386bb2a4d176da245b9"
SEEDVR2_ARCHIVE_SHA256 = "73ebee9da0bfa6876613e23a5f898ebd72ea938eb1d26d9322063172bfb788be"
SEEDVR2_WEIGHTS = ("seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors")  # DiT, VAE
SEEDVR2_SEED = 42  # fixed, so a run is repeatable
# Input pixels per tile side. Each tile is generated on its own, so fewer, larger
# tiles are better and faster. A 512 px texture is one tile and a 1024 px one is
# four. Measured on a 16 GB card: 640 px patches peak at 10.8 GiB, 896 px fail.
SEEDVR2_TILE = 576
# Upstream's VAE works in tiles of this size, in output pixels. On a whole large
# patch it needs a single 4 GiB allocation, which the Intel driver refuses even
# with the memory free.
SEEDVR2_VAE_TILE = (512, 512)
SEEDVR2_VAE_OVERLAP = (64, 64)

SUMMARIES = {
    SEEDVR2: "SeedVR2 3B (one-step diffusion, Apache-2.0), most detail, invents detail, needs 13 GiB of VRAM",
    **{name: model.summary for name, model in MODELS.items()},
}
DEFAULT_MODEL = SEEDVR2
