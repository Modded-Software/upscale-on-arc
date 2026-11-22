"""4x texture upscaling on Intel Arc GPUs (PyTorch XPU)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from arc_upscale.catalog import DEFAULT_MODEL, MODELS, SEEDVR2, SEEDVR2_TILE, SEEDVR2_WEIGHTS, SUMMARIES
from arc_upscale.errors import UserError
from arc_upscale.fetch import fetch_model, fetch_seedvr2, sha256_of
from arc_upscale.files import find_images, plan_outputs
from arc_upscale.gpus import list_gpus, pick_gpu
from arc_upscale.upscalers import SeedVR2Upscaler, Upscaler, XpuUpscaler
from arc_upscale.upscaling import DEFAULT_TILE, TILE_OVERLAP, upscale_image


def is_out_of_memory(error: RuntimeError) -> bool:
    """PyTorch raises OutOfMemoryError. The Level Zero backend below it raises a plain RuntimeError."""
    return isinstance(error, torch.OutOfMemoryError) or "OUT_OF_HOST_MEMORY" in str(error)


def cmd_run(args: argparse.Namespace) -> None:
    tile = args.tile
    if tile is None:
        tile = SEEDVR2_TILE if args.model == SEEDVR2 else DEFAULT_TILE
    if tile <= 2 * TILE_OVERLAP:
        raise UserError(f"--tile must be larger than {2 * TILE_OVERLAP}")
    images = find_images(args.input)
    gpu = pick_gpu(args.gpu)
    device = torch.device("xpu", gpu.index)
    upscaler: Upscaler
    if args.model == SEEDVR2:
        upscaler, model_name = SeedVR2Upscaler(device), SEEDVR2_WEIGHTS[0]
    else:
        model_file = fetch_model(args.model) if args.model in MODELS else Path(args.model)
        if not model_file.is_file():
            raise UserError(f"model not found: {args.model} (use {', '.join(SUMMARIES)}, or a file path)")
        upscaler, model_name = XpuUpscaler(model_file, device), model_file.name
    print(f"GPU {gpu.index}: {gpu.name}, model {model_name}, {upscaler.scale}x")

    root = args.input if args.input.is_dir() else args.input.parent
    for target, source in plan_outputs(images, root, args.output, upscaler.scale).items():
        start = time.perf_counter()
        try:
            with Image.open(source) as image:
                result = upscale_image(upscaler, image, tile, args.tiling)
        except UnidentifiedImageError as error:
            raise UserError(f"not a readable image: {source}") from error
        except RuntimeError as error:
            if not is_out_of_memory(error):
                raise
            raise UserError(f"the GPU ran out of memory on {source}; use a smaller --tile") from error
        target.parent.mkdir(parents=True, exist_ok=True)
        result.save(target)
        print(f"{source} -> {target}  {result.width}x{result.height}  {time.perf_counter() - start:.1f}s")


def cmd_gpus(_args: argparse.Namespace) -> None:
    for gpu in list_gpus():
        print(f"{gpu.index}: {gpu.name}  {gpu.memory / 2**30:.1f} GiB  pci {gpu.pci or 'unknown'}")


def cmd_fetch(args: argparse.Namespace) -> None:
    for name in args.models or SUMMARIES:
        if name not in SUMMARIES:
            raise UserError(f"unknown model: {name} (use {', '.join(SUMMARIES)})")
        if name == SEEDVR2:
            _code, weights = fetch_seedvr2()
            from src.utils.model_registry import MODEL_REGISTRY

            files = {weights / file: MODEL_REGISTRY[file].sha256 for file in SEEDVR2_WEIGHTS}
        else:
            files = {fetch_model(name): MODELS[name].sha256}
        for path, sha256 in files.items():
            if sha256_of(path) != sha256:
                raise UserError(f"checksum mismatch: {path}; delete it and fetch again")
            print(f"{name}: {path} OK")


def build_parser() -> argparse.ArgumentParser:
    models = "\n".join(f"  {name:8s} {summary}" for name, summary in SUMMARIES.items())
    parser = argparse.ArgumentParser(
        prog="arc-upscale",
        description=__doc__,
        epilog=f"models (downloaded on first use):\n{models}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    run = commands.add_parser("run", help="upscale an image file, or every image under a directory")
    run.add_argument("input", type=Path, help="image file or directory")
    run.add_argument("-o", "--output", type=Path, default=Path("out"), help="output directory (default: ./out)")
    run.add_argument("-g", "--gpu", type=int, default=0, help="GPU index, as listed by 'gpus' (default: 0)")
    run.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"{', '.join(SUMMARIES)}, or a model file (default: {DEFAULT_MODEL})",
    )
    run.add_argument(
        "--tile",
        type=int,
        help="tile size in input pixels; lower it if VRAM runs out "
        f"(default: {SEEDVR2_TILE} for {SEEDVR2}, {DEFAULT_TILE} for other models)",
    )
    run.add_argument("--tiling", action="store_true", help="make opposite edges match, for tileable textures")
    run.set_defaults(handler=cmd_run)

    gpus = commands.add_parser("gpus", help="list Intel GPUs")
    gpus.set_defaults(handler=cmd_gpus)

    fetch = commands.add_parser("fetch", help="download and verify models (default: all)")
    fetch.add_argument("models", nargs="*", metavar="model", help=", ".join(SUMMARIES))
    fetch.set_defaults(handler=cmd_fetch)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except UserError as error:
        print(f"arc-upscale: error: {error}", file=sys.stderr)
        return 1
    return 0
