"""4x texture upscaling on Intel Arc GPUs (PyTorch XPU)."""

from __future__ import annotations

import argparse
import gc
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
from arc_upscale.jobs import Job, Spool, default_queue
from arc_upscale.upscalers import SeedVR2Upscaler, Upscaler, XpuUpscaler
from arc_upscale.upscaling import DEFAULT_TILE, TILE_OVERLAP, upscale_image

POLL_SECONDS = 2.0  # how often an idle worker looks for new jobs


def is_out_of_memory(error: RuntimeError) -> bool:
    """PyTorch raises OutOfMemoryError. The Level Zero backend below it raises a plain RuntimeError."""
    return isinstance(error, torch.OutOfMemoryError) or "OUT_OF_HOST_MEMORY" in str(error)


def pick_tile(model: str, tile: int | None) -> int:
    if tile is None:
        tile = SEEDVR2_TILE if model == SEEDVR2 else DEFAULT_TILE
    if tile <= 2 * TILE_OVERLAP:
        raise UserError(f"--tile must be larger than {2 * TILE_OVERLAP}")
    return tile


def model_not_found(model: str) -> UserError:
    return UserError(f"model not found: {model} (use {', '.join(SUMMARIES)}, or a file path)")


def load_upscaler(model: str, device: torch.device) -> tuple[Upscaler, str]:
    """The model `model` on `device`, and the name of its weights file."""
    if model == SEEDVR2:
        return SeedVR2Upscaler(device), SEEDVR2_WEIGHTS[0]
    model_file = fetch_model(model) if model in MODELS else Path(model)
    if not model_file.is_file():
        raise model_not_found(model)
    return XpuUpscaler(model_file, device), model_file.name


def upscale_files(upscaler: Upscaler, source: Path, output: Path, tile: int, tiling: bool) -> None:
    """Upscale the image `source`, or every image under it, into `output`."""
    images = find_images(source)
    root = source if source.is_dir() else source.parent
    for target, image_path in plan_outputs(images, root, output, upscaler.scale).items():
        start = time.perf_counter()
        try:
            with Image.open(image_path) as image:
                result = upscale_image(upscaler, image, tile, tiling)
        except UnidentifiedImageError as error:
            raise UserError(f"not a readable image: {image_path}") from error
        except RuntimeError as error:
            if not is_out_of_memory(error):
                raise
            raise UserError(f"the GPU ran out of memory on {image_path}; use a smaller --tile") from error
        target.parent.mkdir(parents=True, exist_ok=True)
        result.save(target)
        print(f"{image_path} -> {target}  {result.width}x{result.height}  {time.perf_counter() - start:.1f}s")


def cmd_run(args: argparse.Namespace) -> None:
    tile = pick_tile(args.model, args.tile)
    find_images(args.input)  # report a bad input before the model loads
    gpu = pick_gpu(args.gpu)
    upscaler, model_name = load_upscaler(args.model, torch.device("xpu", gpu.index))
    print(f"GPU {gpu.index}: {gpu.name}, model {model_name}, {upscaler.scale}x")
    upscale_files(upscaler, args.input, args.output, tile, args.tiling)


# ---------------------------------------------------------------- queue


class ModelSlot:
    """The one model a worker keeps loaded. A job for another model replaces it, as two may not fit in VRAM."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model: str | None = None
        self.upscaler: Upscaler | None = None

    def get(self, model: str) -> Upscaler:
        if self.upscaler is not None and self.model == model:
            return self.upscaler
        if self.upscaler is not None:
            self.model, self.upscaler = None, None
            gc.collect()
            if self.device.type == "xpu":
                torch.xpu.empty_cache()
        upscaler, model_name = load_upscaler(model, self.device)
        print(f"model {model_name}, {upscaler.scale}x")
        self.model, self.upscaler = model, upscaler
        return upscaler


def open_spool(args: argparse.Namespace) -> Spool:
    return Spool(args.queue if args.queue is not None else default_queue())


def cmd_submit(args: argparse.Namespace) -> None:
    model = args.model
    if model not in SUMMARIES:
        if not Path(model).is_file():
            raise model_not_found(model)
        model = str(Path(model).resolve())
    pick_tile(model, args.tile)
    find_images(args.input)  # report a bad input now, not later on a worker
    job = Job(input=args.input.resolve(), output=args.output.resolve(), model=model, tile=args.tile, tiling=args.tiling)
    path = open_spool(args).submit(job)
    print(f"queued {path.name}: {job.input} -> {job.output}")


def cmd_worker(args: argparse.Namespace) -> None:
    gpu = pick_gpu(args.gpu)
    spool = open_spool(args)
    slot = ModelSlot(torch.device("xpu", gpu.index))
    with spool.worker_lock(gpu.index):
        for path in spool.requeue(gpu.index):
            print(f"requeued {path.name}: an earlier worker on GPU {gpu.index} stopped before it was done")
        print(f"GPU {gpu.index}: {gpu.name}, worker on {spool.root}")
        while True:
            claimed = spool.claim(gpu.index)
            if claimed is None:
                if args.until_empty:
                    print("queue is empty")
                    return
                time.sleep(POLL_SECONDS)
                continue
            job = claimed.job
            print(f"job {claimed.path.name}: {job.input} -> {job.output}, model {job.model}")
            try:
                upscale_files(slot.get(job.model), job.input, job.output, pick_tile(job.model, job.tile), job.tiling)
            except UserError as error:
                target = spool.fail(claimed, str(error))
                print(f"arc-upscale: job failed: {error} (see {target})", file=sys.stderr)
            else:
                print(f"job done: {spool.finish(claimed)}")


def cmd_queue(args: argparse.Namespace) -> None:
    spool = open_spool(args)
    try:
        pending, done, failed = (spool.jobs(directory) for directory in (spool.pending, spool.done, spool.failed))
        working = {directory.name: spool.jobs(directory) for directory in spool.workers()}
    except FileNotFoundError as error:
        raise UserError(f"a job moved while the queue was read ({error.filename}); run the command again") from error
    print(f"queue {spool.root}: {len(pending)} pending, {len(done)} done, {len(failed)} failed")
    for worker, claimed in working.items():
        for entry in claimed:
            print(f"  {worker}: {entry.path.name}  {entry.job.input}")
    for entry in failed:
        print(f"  failed: {entry.path.name}  {entry.job.input}: {entry.job.error}")


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


def add_job_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="image file or directory")
    parser.add_argument("-o", "--output", type=Path, default=Path("out"), help="output directory (default: ./out)")
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"{', '.join(SUMMARIES)}, or a model file (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--tile",
        type=int,
        help="tile size in input pixels; lower it if VRAM runs out "
        f"(default: {SEEDVR2_TILE} for {SEEDVR2}, {DEFAULT_TILE} for other models)",
    )
    parser.add_argument("--tiling", action="store_true", help="make opposite edges match, for tileable textures")


def add_gpu_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU index, as listed by 'gpus' (default: 0)")


def add_queue_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--queue", type=Path, help="queue directory (default: queue/ in the project)")


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
    add_job_arguments(run)
    add_gpu_argument(run)
    run.set_defaults(handler=cmd_run)

    submit = commands.add_parser("submit", help="add a job to the queue, for a worker to run")
    add_job_arguments(submit)
    add_queue_argument(submit)
    submit.set_defaults(handler=cmd_submit)

    worker = commands.add_parser("worker", help="run queued jobs on one GPU, keeping the model loaded")
    add_gpu_argument(worker)
    add_queue_argument(worker)
    worker.add_argument("--until-empty", action="store_true", help="stop when no job is left, instead of waiting")
    worker.set_defaults(handler=cmd_worker)

    status = commands.add_parser("queue", help="show pending, running, done and failed jobs")
    add_queue_argument(status)
    status.set_defaults(handler=cmd_queue)

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
