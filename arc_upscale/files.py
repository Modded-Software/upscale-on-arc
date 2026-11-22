"""Input images and the output file each one is written to."""

from __future__ import annotations

from pathlib import Path

from arc_upscale.errors import UserError

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tif", ".tiff", ".webp"}


def find_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise UserError(f"input does not exist: {source}")
    images = sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise UserError(f"no images found under {source}")
    return images


def plan_outputs(images: list[Path], root: Path, output: Path, scale: int) -> dict[Path, Path]:
    """Map each output file to its input, keeping the layout below `root`.

    Outputs are always PNG, so wall.png and wall.jpg would collide. That is an error.
    """
    plan: dict[Path, Path] = {}
    for image in images:
        target = output / image.relative_to(root).with_name(f"{image.stem}_{scale}x.png")
        if target in plan:
            raise UserError(f"{plan[target]} and {image} would both be written to {target}")
        plan[target] = image
    return plan
