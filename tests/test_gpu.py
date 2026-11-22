"""Tests with the real models on a real GPU. Choose the GPU with: pytest --gpu N"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from arc_upscale import catalog, cli, fetch, gpus, upscalers, upscaling

ROOT = Path(os.environ[catalog.ROOT_VARIABLE])  # set by `make test`
TEXTURES = sorted((ROOT / "testdata" / "gt").glob("*.png"))
CUTOUT = ROOT / "testdata" / "alpha" / "brick_cutout.png"
LOW = 256  # the 1024px ground truth is reduced to this, then upscaled back


@pytest.fixture(scope="session", params=list(catalog.SUMMARIES))
def upscaler(request: pytest.FixtureRequest, gpu: gpus.Gpu) -> upscalers.Upscaler:
    device = torch.device("xpu", gpu.index)
    if request.param == catalog.SEEDVR2:
        return upscalers.SeedVR2Upscaler(device)
    return upscalers.XpuUpscaler(fetch.fetch_model(request.param), device)


def invents_detail(upscaler: upscalers.Upscaler) -> bool:
    return isinstance(upscaler, upscalers.SeedVR2Upscaler)


def load(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def low_res(image: Image.Image) -> Image.Image:
    return image.resize((LOW, LOW), Image.Resampling.LANCZOS)


def pixels(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float64)


def psnr(a: Image.Image, b: Image.Image) -> float:
    return float(10 * np.log10(255.0**2 / np.mean((pixels(a) - pixels(b)) ** 2)))


def sharpness(image: Image.Image) -> float:
    """Mean gradient magnitude."""
    return float(np.mean(np.hypot(*np.gradient(pixels(image.convert("L"))))))


def wrap_ratio(image: Image.Image) -> float:
    """Pixel step across the wrap-around edges, relative to the step between neighbours inside.

    About 1 for a texture that tiles; well above 1 for a visible seam.
    """
    data = pixels(image)
    ratios = []
    for axis in (0, 1):
        interior = np.abs(np.diff(data, axis=axis)).mean()
        across = np.abs(np.take(data, 0, axis=axis) - np.take(data, -1, axis=axis)).mean()
        ratios.append(across / interior)
    return float(max(ratios))


@pytest.mark.parametrize("texture", TEXTURES, ids=lambda path: path.stem)
def test_output_is_sharper_than_bicubic_and_stays_close(upscaler: upscalers.Upscaler, texture: Path) -> None:
    truth = load(texture)
    low = low_res(truth)
    bicubic = low.resize(truth.size, Image.Resampling.BICUBIC)
    result = upscaling.upscale_image(upscaler, low)
    assert result.size == truth.size
    assert sharpness(result) > 1.15 * sharpness(bicubic)
    # A diffusion model invents detail, so it lands further from the original than
    # bicubic does: 2.8 dB further on the brick wall. It must stay in the same range.
    margin = 5.0 if invents_detail(upscaler) else 2.0
    assert psnr(result, truth) > psnr(bicubic, truth) - margin


@pytest.mark.parametrize("texture", TEXTURES, ids=lambda path: path.stem)
def test_tiled_result_matches_single_pass(upscaler: upscalers.Upscaler, texture: Path) -> None:
    if invents_detail(upscaler):
        pytest.skip("a diffusion model renders each tile on its own: 27 to 28 dB against a single pass")
    low = low_res(load(texture))
    single = upscaling.upscale_image(upscaler, low, tile=LOW)
    tiled = upscaling.upscale_image(upscaler, low, tile=96)
    assert psnr(tiled, single) > 40.0


@pytest.mark.parametrize("texture", TEXTURES, ids=lambda path: path.stem)
def test_tiling_removes_the_seam(upscaler: upscalers.Upscaler, texture: Path) -> None:
    low = low_res(load(texture))
    plain = wrap_ratio(upscaling.upscale_image(upscaler, low))
    tiling = wrap_ratio(upscaling.upscale_image(upscaler, low, tiling=True))
    assert tiling < 1.5
    assert tiling < plain


def test_cutout_keeps_its_alpha(upscaler: upscalers.Upscaler) -> None:
    with Image.open(CUTOUT) as image:
        source = image.convert("RGBA")
    result = upscaling.upscale_image(upscaler, source)
    assert result.mode == "RGBA"
    assert result.size == (source.width * 4, source.height * 4)
    expected = source.getchannel("A").resize(result.size, Image.Resampling.BICUBIC)
    assert psnr(result.getchannel("A"), expected) > 20.0


def engine_busy_ns() -> dict[str, int]:
    """GPU engine busy time of this process per PCI address, from the kernel's DRM fdinfo.

    This does not involve PyTorch, so it checks PyTorch's device numbering independently.
    """
    clients: dict[tuple[str, str], int] = {}
    # The listing is taken first: the descriptor that read it is in it, and is closed by now.
    for entry in list(Path("/proc/self/fdinfo").iterdir()):
        if not entry.exists():
            continue
        text = entry.read_text()
        fields = dict(line.split(":", 1) for line in text.splitlines() if ":" in line)
        if "drm-pdev" not in fields:
            continue
        busy = sum(
            int(value.strip().removesuffix("ns"))
            for key, value in fields.items()
            if key.startswith("drm-engine-") and not key.startswith("drm-engine-capacity-")
        )
        clients[fields["drm-pdev"].strip(), fields["drm-client-id"].strip()] = busy
    totals: dict[str, int] = {}
    for (pci, _client), busy in clients.items():
        totals[pci] = totals.get(pci, 0) + busy
    return totals


def test_work_runs_on_the_selected_gpu(upscaler: upscalers.Upscaler, gpu: gpus.Gpu) -> None:
    low = low_res(load(TEXTURES[0]))
    before = engine_busy_ns()
    if gpu.pci not in before:
        pytest.skip(f"the kernel driver reports no engine busy time for GPU {gpu.index} (pci {gpu.pci})")
    upscaling.upscale_image(upscaler, low)
    after = engine_busy_ns()
    used = {pci: after[pci] - before.get(pci, 0) for pci in after}
    assert used[gpu.pci] > 0
    assert all(busy == 0 for pci, busy in used.items() if pci != gpu.pci), used


def test_unreadable_image_is_reported(tmp_path: Path, gpu: gpus.Gpu, capsys: pytest.CaptureFixture[str]) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a png")
    assert cli.main(["run", "-g", str(gpu.index), "-m", "fast", "-o", str(tmp_path / "out"), str(broken)]) == 1
    assert f"arc-upscale: error: not a readable image: {broken}\n" in capsys.readouterr().err


def run_checked(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command; a non-zero exit raises CalledProcessError, which holds stdout and stderr."""
    return subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, check=True)


def expect_success(
    command: list[str], cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a command that must exit 0. On failure, the test fails with the command's stderr."""
    try:
        return run_checked(command, cwd, env)
    except subprocess.CalledProcessError as failed:
        raise AssertionError(f"exit {failed.returncode}: {failed.stderr}") from failed


def expect_failure(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CalledProcessError:
    """Run a command that must exit non-zero, and return the error with its output."""
    with pytest.raises(subprocess.CalledProcessError) as failed:
        run_checked(command, cwd, env)
    return failed.value


def fresh(arguments: list[str]) -> list[str]:
    """upscale.py in a process of its own, so that the weights load anew."""
    return [sys.executable, str(ROOT / "upscale.py"), *arguments]


def test_seedvr2_default_run_on_a_whole_texture_is_sound(tmp_path: Path, gpu: gpus.Gpu) -> None:
    """A 1024 px texture in tiling mode, with the default tile, as the first work of a process.

    A large first patch once gave corrupt output without an error, because GPU
    memory ran short while the weights loaded. A corrupt result is far from a
    bicubic upscale: 3 to 14 dB, against 28 dB for a sound one.
    """
    source = load(TEXTURES[0])
    source.save(tmp_path / "wall.png")
    expect_success(fresh(["run", "-g", str(gpu.index), "--tiling", "-o", "result", "wall.png"]), tmp_path)
    result = load(tmp_path / "result" / "wall_4x.png")
    assert result.size == (source.width * 4, source.height * 4)
    assert psnr(result, source.resize(result.size, Image.Resampling.BICUBIC)) > 20.0
    assert wrap_ratio(result) < 1.5


def test_seedvr2_too_large_a_tile_is_reported(tmp_path: Path, gpu: gpus.Gpu) -> None:
    load(TEXTURES[0]).save(tmp_path / "wall.png")
    failed = expect_failure(fresh(["run", "-g", str(gpu.index), "--tile", "1024", "wall.png"]), tmp_path)
    assert failed.returncode == 1
    assert "arc-upscale: error: the GPU ran out of memory on wall.png; use a smaller --tile" in failed.stderr
    assert "Traceback" not in failed.stderr


# ---------------------------------------------------------------- the installed command


@pytest.fixture
def command(tmp_path: Path) -> list[str]:
    """The launcher, reached through a symlink in another directory, as after `make install`."""
    link = tmp_path / "bin" / "arc-upscale"
    link.parent.mkdir()
    link.symlink_to(ROOT / "arc-upscale")
    return [str(link)]


def launcher_env(root: Path | None = None) -> dict[str, str]:
    """The environment with ARC_UPSCALE_ROOT unset, or set to `root`."""
    environment = {key: value for key, value in os.environ.items() if key != "ARC_UPSCALE_ROOT"}
    if root is not None:
        environment["ARC_UPSCALE_ROOT"] = str(root)
    return environment


def test_project_location_can_be_overridden(tmp_path: Path, gpu: gpus.Gpu) -> None:
    copy = tmp_path / "arc-upscale"
    copy.write_bytes((ROOT / "arc-upscale").read_bytes())
    done = expect_success(["bash", str(copy), "gpus"], tmp_path, launcher_env(ROOT))
    assert f"{gpu.index}: {gpu.name}" in done.stdout


def test_command_lists_the_gpu(command: list[str], tmp_path: Path, gpu: gpus.Gpu) -> None:
    done = expect_success([*command, "gpus"], tmp_path, launcher_env())
    assert f"{gpu.index}: {gpu.name}" in done.stdout


def test_command_upscales_a_directory_from_elsewhere(command: list[str], tmp_path: Path, gpu: gpus.Gpu) -> None:
    (tmp_path / "in" / "sub").mkdir(parents=True)
    low_res(load(TEXTURES[0])).save(tmp_path / "in" / "sub" / "wall.png")
    run = [*command, "run", "-g", str(gpu.index), "-m", "fast", "-o", "result", "in"]
    done = expect_success(run, tmp_path, launcher_env())
    assert f"GPU {gpu.index}:" in done.stdout
    with Image.open(tmp_path / "result" / "sub" / "wall_4x.png") as image:
        assert image.size == (LOW * 4, LOW * 4)


def test_command_reports_a_missing_gpu(command: list[str], tmp_path: Path) -> None:
    low_res(load(TEXTURES[0])).save(tmp_path / "wall.png")
    failed = expect_failure([*command, "run", "-g", "99", "wall.png"], tmp_path, launcher_env())
    assert failed.returncode == 1
    assert "GPU 99 does not exist" in failed.stderr
    assert "Traceback" not in failed.stderr
