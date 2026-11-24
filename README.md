# upscale-on-arc

4x upscaling of game textures on Intel Arc GPUs, using PyTorch's XPU backend.

## Install

You need Linux, an Intel Arc GPU with the Level Zero driver, and [uv](https://docs.astral.sh/uv/).
On Ubuntu the driver is `sudo apt install libze1 libze-intel-gpu1`.

```bash
make install    # uv sync --frozen, then links arc-upscale into ~/.local/bin; set BINDIR to change that
```

Or skip this and call the script by its path. It works from any directory.

The first run installs the Python environment (about 7.4 GB, in `.venv/`) and downloads the model it needs (into `models/`). The default model is 6.8 GB. That takes a while. Downloads are checksum-verified.

## Use

```bash
arc-upscale gpus                                # list GPUs
arc-upscale run textures/ -o textures_4x/       # every image under a directory
arc-upscale run -g 1 --tiling wall.png          # one tileable texture, on GPU 1
arc-upscale run -m pbrify textures/             # a faster model that invents nothing
arc-upscale fetch                               # download all models ahead of time
arc-upscale run --help                          # all options
```

- Output is 8-bit PNG, named `<name>_4x.png`, in `./out` unless `-o` is given. The input's directory layout is kept.
- Transparency is kept.
- `--tiling` makes opposite edges match, for textures that tile.
- `--tile N` sets the tile size. Lower it if the GPU runs out of memory.
- `-g N` picks the GPU by the number that `gpus` prints. The default is 0.
- One run uses one GPU: see "SeedVR2 and VRAM". To use several, queue the work (below).

## Queue

```bash
arc-upscale worker -g 0 &                       # one worker per GPU; it waits for jobs
arc-upscale worker -g 1 &
arc-upscale submit textures/ -o textures_4x/    # takes the same options as run
arc-upscale submit -m pbrify --tiling wall.png
arc-upscale queue                               # pending, running, done and failed jobs
```

- A worker keeps its model loaded between jobs. A job for another model replaces it.
- Each job is a JSON file in `queue/` in the project. `--queue DIR` picks another directory, for every command that uses it.
- `submit` checks the input and the model, and stores absolute paths. Workers run in any directory.
- A job that fails, for example on an unreadable image or with too little VRAM, goes to `queue/failed/` with its error. The worker goes on with the next one. Any other error stops the worker.
- A worker that stops leaves its job in `queue/working/gpu<N>/`. The next worker on that GPU puts it back in the queue first. A job that crashes the worker every time would loop, so it needs to be removed by hand.
- `worker --until-empty` stops when no job is left.
- Done jobs stay in `queue/done/` until you delete them.
- Workers claim jobs by renaming files, so the queue must be on a local filesystem on the same host.

## Models

| Name | Model | Notes |
|---|---|---|
| `seedvr2` (default) | [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) 3B, fp16 (ByteDance, Apache-2.0) | One-step diffusion. Most detail. Slow. Invents detail. |
| `pbrify` | 4x-PBRify_UpscalerV4 (DAT2, CC0) | Stays close to the input. Faster. |
| `fast` | 4x-PBRify_RPLKSRd_V3 (RealPLKSR, CC0) | Fastest, softer. |

The PBRify models are by [Kim2091](https://github.com/Kim2091/Kim2091-Models), trained for game textures, released in May 2025.
`-m` also takes the path of any other single-image model that [spandrel](https://github.com/chaiNNer-org/spandrel) can load.

**How they were chosen:** by public signals only: release date, published scores, download counts, user reports, and whether the project is maintained. **Nothing was benchmarked here.** "Best" is an informed pick, not a measured result.

- SeedVR2 is the upscaler named most often in 2026 "best upscaler" threads, and has by far the most downloads (about 790,000 for the weights used here). Its paper is at ICLR 2026. The weights are from June 2025.
- It is a general image and video model, not a texture model. No FOSS texture-specific upscaler from 2026 was found. PBRify V4 is the newest one with open weights.
- Not FOSS, so excluded: NVIDIA PiD, 4x-UltraSharpV2, Ubisoft CHORD, UltraTex-SR.

### What to know about SeedVR2

- **It invents detail.** The result looks sharp and plausible, but it is not the original. On one of the two test textures it was 2.8 dB further from the original than a plain bicubic resize. Use `pbrify` where fidelity matters, for example for normal maps or data textures.
- **Images above 576 px are rendered tile by tile.** Each tile is generated on its own, then neighbours are cross-faded. No seam was visible on the test texture, but two tiles do not agree on detail: 96 px tiles matched a single pass at only 27 to 28 dB. So the default tile is large: a 512 px texture is one tile, a 1024 px one is four.
- **Transparency** goes through the model as a grey image. That was measured to help with the PBRify models. It was not measured for SeedVR2.
- Upstream prints `Memory check failed: No GPU backend available (CUDA/MPS)` at start. Ignore it. Upstream does not know about Intel GPUs; the work does run on the GPU.

### SeedVR2 and VRAM

Measured on one Arc A770 16 GB (15.1 GiB usable), by patch size in input pixels:

| Patch | Peak allocated | Peak reserved | |
|---|---|---|---|
| 288 | 8.6 GiB | 9.5 GiB | |
| 544 | 9.7 GiB | 11.4 GiB | |
| 640 | 10.8 GiB | 12.6 GiB | covers the default: 576, or 608 with `--tiling` |
| 768 | 12.5 GiB | 14.7 GiB | works, but with no margin |
| 896 | | | fails: out of memory |

- The limit is the diffusion model's working memory. It cannot be split over two cards. Moving the VAE to a second card was tried: it freed 0.5 GiB and allowed no larger tile, so there is no option for it.
- The VAE works in 512 px tiles. On a whole 2048 px patch it needs a single 4 GiB allocation, which the Intel driver refuses even with 15 GiB free. With and without VAE tiles the output differed by 1.5 of 255 on average, with no difference along the tile lines.
- A card that also drives a desktop has less free memory. Lower `--tile` there.

## What was measured

On one Arc A770 16 GB (Ubuntu 24.04, `i915` driver, PyTorch 2.15.0+xpu). Seconds per image for the whole command step, including reading and writing the files, after warm-up. One run each.

| Model | 256 px input | 512 px | 1024 px |
|---|---|---|---|
| `seedvr2` | 3.8 | 12.9 | 61.9 |
| `seedvr2 --tiling` | 4.4 | 14.2 | 71.3 |
| `pbrify` | 1.4 | 10.8 | 31.4 |
| `pbrify --tiling` | 1.7 | 6.7 | 26.5 |
| `fast` | 0.4 | 2.6 | 8.3 |
| `fast --tiling` | 0.5 | 2.0 | 7.6 |

With `seedvr2` the first image of a run takes about 9 s longer, as the weights load. Runs are repeatable: the seed is fixed, and the same input gave the same file.

The tests check that output is sharper than bicubic, stays in range of the original, tiles without a seam in tiling mode, and that the work ran on the selected GPU.
This is a basic check on two Poly Haven textures. It is not a quality benchmark, and those textures may be in the models' training data.

## Development

```bash
make test GPU=1    # pytest; the GPU tests run on the given GPU
make lint          # shellcheck and ruff
make clean         # remove ./out in the project
```

`arc-upscale` finds the project from its own location, following symlinks. Set `ARC_UPSCALE_ROOT` to point it at a different checkout.
The launcher exports the project directory as `ARC_UPSCALE_ROOT`. The package reads `models/` from there and nowhere else, and stops with an error if the variable is unset. `make test` sets it too.

The code is the `arc_upscale` package. `upscale.py` is only the entry point that `arc-upscale` runs.

| Module | Holds |
|---|---|
| `catalog` | The named models, with pinned files, commits and checksums |
| `fetch` | Checksum-verified downloads |
| `gpus` | GPU listing and PCI addresses |
| `upscalers` | A model loaded onto one GPU, called per patch |
| `upscaling` | Tiles, tiling mode and alpha, for whole images |
| `files` | Input discovery and output names |
| `jobs` | The job queue, in a spool directory |
| `cli` | The commands |

SeedVR2's code is not a Python package. `arc_upscale/catalog.py` pins one upstream commit, and `fetch` downloads that source archive, checksum-verified, into `models/`. Upstream has no Intel GPU support. Three things were needed, all in `SeedVR2Upscaler`:

- Pass the `xpu` device straight to upstream's pipeline functions. That works unchanged.
- Free PyTorch's allocator cache after each phase. Upstream does that for CUDA and MPS only. Without it, reserved memory climbed past the card's 15.1 GiB. The output was then corrupt, with no error, and differed from run to run.
- Turn on upstream's VAE tiling, for the driver's 4 GiB allocation limit.

As a guard, output with invalid values is refused, not written.

## Licence

SeedVR2's code and weights are Apache-2.0. The PBRify models and the test textures (Poly Haven) are CC0. The code in this project has no licence yet.
