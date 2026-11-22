"""Tests that need no GPU. The model is replaced by nearest-neighbour scaling."""

import hashlib
import itertools
import struct
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from arc_upscale import catalog, cli, fetch, files, gpus, upscalers, upscaling
from arc_upscale.catalog import ModelFile
from arc_upscale.errors import UserError
from arc_upscale.upscaling import TILE_OVERLAP


class NearestUpscaler:
    """Stands in for a model. Every output pixel depends on one input pixel only,

    so a correct tiled result equals a single pass exactly.
    """

    scale = 4

    def __init__(self) -> None:
        self.patch_sizes: list[tuple[int, int]] = []

    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        self.patch_sizes.append((patch.shape[2], patch.shape[3]))
        return patch.repeat_interleave(self.scale, dim=2).repeat_interleave(self.scale, dim=3)


def random_image(height: int, width: int) -> torch.Tensor:
    return torch.rand(1, 3, height, width, generator=torch.Generator().manual_seed(0))


# ---------------------------------------------------------------- tiling


@pytest.mark.parametrize("tile", [33, 64, 256])
@pytest.mark.parametrize("length", [1, 32, 33, 64, 65, 100, 255, 256, 257, 500, 1024, 1500])
def test_tiles_cover_the_image_and_overlap(length: int, tile: int) -> None:
    starts = upscaling.tile_starts(length, tile)
    assert starts[0] == 0
    assert starts == sorted(set(starts))
    assert starts[-1] + tile >= length
    if length > tile:
        assert starts[-1] + tile == length
    for left, right in itertools.pairwise(starts):
        assert left + tile - right >= TILE_OVERLAP


def test_feather_ramps_only_towards_neighbours() -> None:
    assert torch.equal(upscaling.feather(100, 16, at_start=True, at_end=True), torch.ones(100))
    weights = upscaling.feather(100, 16, at_start=False, at_end=True)
    assert (weights[:16].diff() > 0).all()
    assert torch.equal(weights[16:], torch.ones(84))
    weights = upscaling.feather(100, 16, at_start=True, at_end=False)
    assert torch.equal(weights[:84], torch.ones(84))
    assert (weights[84:].diff() < 0).all()


@pytest.mark.parametrize("length", [1, 2, 3, 31, 32, 100])
def test_feather_weights_are_never_zero(length: int) -> None:
    assert (upscaling.feather(length, 16, at_start=False, at_end=False) > 0).all()


@pytest.mark.parametrize("size", [(40, 40), (64, 64), (65, 200), (300, 130)])
def test_tiled_result_equals_single_pass(size: tuple[int, int]) -> None:
    image = random_image(*size)
    fake = NearestUpscaler()
    result = upscaling.upscale_tiled(fake, image, tile=64)
    assert torch.allclose(result, fake(image), atol=1e-6)


def test_tiles_are_never_larger_than_requested() -> None:
    fake = NearestUpscaler()
    upscaling.upscale_tiled(fake, random_image(300, 130), tile=64)
    assert len(fake.patch_sizes) > 1
    assert all(height <= 64 and width <= 64 for height, width in fake.patch_sizes)


# ---------------------------------------------------------------- tiling


@pytest.mark.parametrize("size", [(8, 8), (64, 64), (100, 300)])
def test_tiling_keeps_size_and_content(size: tuple[int, int]) -> None:
    image = random_image(*size)
    fake = NearestUpscaler()
    result = upscaling.upscale_tensor(fake, image, tile=64, tiling=True)
    assert torch.allclose(result, fake(image), atol=1e-6)


def test_tiling_image_that_fits_one_tile_is_one_model_call() -> None:
    fake = NearestUpscaler()
    upscaling.upscale_tensor(fake, random_image(64, 64), tile=64, tiling=True)
    assert len(fake.patch_sizes) == 1


def test_tiling_shows_the_model_the_opposite_edge() -> None:
    image = random_image(64, 64)
    seen: list[torch.Tensor] = []

    class Recorder(NearestUpscaler):
        def __call__(self, patch: torch.Tensor) -> torch.Tensor:
            seen.append(patch)
            return super().__call__(patch)

    upscaling.upscale_tensor(Recorder(), image, tile=64, tiling=True)
    pad = upscaling.TILING_PAD
    assert torch.equal(seen[0][:, :, pad:-pad, :pad], image[:, :, :, -pad:])
    assert torch.equal(seen[0][:, :, :pad, pad:-pad], image[:, :, -pad:, :])


def test_tiling_edges_match_for_a_model_that_renders_by_position() -> None:
    """Such a model renders the wrapped border and the real edge differently, as a diffusion model does."""

    class Shaded(NearestUpscaler):
        def __call__(self, patch: torch.Tensor) -> torch.Tensor:
            out = super().__call__(patch)
            across = torch.linspace(0, 0.4, out.shape[3])[None, None, None, :]
            down = torch.linspace(0, 0.4, out.shape[2])[None, None, :, None]
            return out + across + down

    image = torch.full((1, 3, 64, 64), 0.1)
    result = upscaling.upscale_tensor(Shaded(), image, tile=64, tiling=True)
    assert result.shape == (1, 3, 256, 256)
    inside = result.diff(dim=3).abs().max()
    assert (result[..., 0] - result[..., -1]).abs().max() <= 2 * inside
    assert (result[..., 0, :] - result[..., -1, :]).abs().max() <= 2 * inside
    # Without the blend, the edges would differ by most of the shading range.
    cut = Shaded()(torch.nn.functional.pad(image, (16, 16, 16, 16), mode="circular"))[..., 64:-64, 64:-64]
    assert (cut[..., 0] - cut[..., -1]).abs().max() > 0.25


# ---------------------------------------------------------------- images


def nearest(image: Image.Image) -> Image.Image:
    return image.resize((image.width * 4, image.height * 4), Image.Resampling.NEAREST)


def test_rgb_image() -> None:
    source = Image.effect_noise((40, 24), 64).convert("RGB")
    result = upscaling.upscale_image(NearestUpscaler(), source)
    assert result.mode == "RGB"
    assert np.array_equal(np.asarray(result), np.asarray(nearest(source)))


def test_greyscale_image_becomes_rgb() -> None:
    source = Image.effect_noise((40, 24), 64)
    result = upscaling.upscale_image(NearestUpscaler(), source)
    assert result.mode == "RGB"
    assert np.array_equal(np.asarray(result), np.asarray(nearest(source.convert("RGB"))))


def test_transparency_is_kept() -> None:
    source = Image.effect_noise((40, 24), 64).convert("RGB")
    source.putalpha(Image.linear_gradient("L").resize((40, 24)))
    result = upscaling.upscale_image(NearestUpscaler(), source)
    assert result.mode == "RGBA"
    assert np.array_equal(np.asarray(result), np.asarray(nearest(source)))


def test_opaque_alpha_stays_opaque_without_a_model_call() -> None:
    source = Image.new("RGBA", (40, 24), (10, 20, 30, 255))
    fake = NearestUpscaler()
    result = upscaling.upscale_image(fake, source)
    assert result.mode == "RGBA"
    assert result.getextrema()[3] == (255, 255)
    assert len(fake.patch_sizes) == 1


def test_palette_transparency_is_kept() -> None:
    source = Image.new("P", (40, 24), 0)
    source.putpalette([255, 0, 0, 0, 255, 0])
    source.paste(1, (0, 0, 20, 24))
    source.info["transparency"] = 0
    result = upscaling.upscale_image(NearestUpscaler(), source)
    assert result.mode == "RGBA"
    assert result.getpixel((0, 0)) == (0, 255, 0, 255)
    assert result.getpixel((159, 0)) == (255, 0, 0, 0)


# ---------------------------------------------------------------- files


def test_outputs_keep_the_directory_layout(tmp_path: Path) -> None:
    root, out = tmp_path / "in", tmp_path / "out"
    images = [root / "wall.png", root / "sub" / "floor.jpg"]
    plan = files.plan_outputs(images, root, out, scale=4)
    assert plan == {out / "wall_4x.png": images[0], out / "sub" / "floor_4x.png": images[1]}


def test_two_inputs_with_one_output_name_are_refused(tmp_path: Path) -> None:
    with pytest.raises(UserError, match="would both be written"):
        files.plan_outputs([tmp_path / "wall.png", tmp_path / "wall.jpg"], tmp_path, tmp_path / "out", scale=4)


def test_find_images(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    for name in ("b.PNG", "a.jpg", "notes.txt", "sub/c.webp"):
        (tmp_path / name).touch()
    assert files.find_images(tmp_path) == [tmp_path / "a.jpg", tmp_path / "b.PNG", tmp_path / "sub" / "c.webp"]
    assert files.find_images(tmp_path / "a.jpg") == [tmp_path / "a.jpg"]


def test_find_images_reports_bad_input(tmp_path: Path) -> None:
    with pytest.raises(UserError, match="does not exist"):
        files.find_images(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(UserError, match="no images"):
        files.find_images(tmp_path / "empty")


# ---------------------------------------------------------------- PCI address


def level_zero_uuid(vendor: int, device: int, domain: int, bus: int, dev: int, func: int) -> bytes:
    return struct.pack("<4H3B5x", vendor, device, 8, domain, bus, dev, func)


def fake_sysfs(root: Path, address: str, vendor: int, device: int) -> Path:
    (root / address).mkdir(parents=True)
    (root / address / "vendor").write_text(f"0x{vendor:04x}\n")
    (root / address / "device").write_text(f"0x{device:04x}\n")
    return root


def test_pci_address_from_uuid(tmp_path: Path) -> None:
    sysfs = fake_sysfs(tmp_path, "0000:09:00.0", 0x8086, 0x56A0)
    assert gpus.pci_address(level_zero_uuid(0x8086, 0x56A0, 0, 9, 0, 0), sysfs) == "0000:09:00.0"


def test_pci_address_is_none_when_sysfs_disagrees(tmp_path: Path) -> None:
    sysfs = fake_sysfs(tmp_path, "0000:09:00.0", 0x8086, 0x56A0)
    assert gpus.pci_address(level_zero_uuid(0x8086, 0x56A1, 0, 9, 0, 0), sysfs) is None
    assert gpus.pci_address(level_zero_uuid(0x8086, 0x56A0, 0, 3, 0, 0), sysfs) is None


# ---------------------------------------------------------------- model download


def served_model(tmp_path: Path, sha256: str | None = None) -> dict[str, ModelFile]:
    source = tmp_path / "served.bin"
    source.write_bytes(b"model weights")
    digest = sha256 or hashlib.sha256(b"model weights").hexdigest()
    return {"demo": ModelFile(file="demo.bin", url=source.as_uri(), sha256=digest, summary="demo")}


def test_model_is_downloaded_once(tmp_path: Path) -> None:
    models, directory = served_model(tmp_path), tmp_path / "models"
    path = fetch.fetch_model("demo", models, directory)
    assert path == directory / "demo.bin"
    assert path.read_bytes() == b"model weights"
    (tmp_path / "served.bin").unlink()
    assert fetch.fetch_model("demo", models, directory) == path


def test_download_with_wrong_checksum_is_discarded(tmp_path: Path) -> None:
    models, directory = served_model(tmp_path, sha256="0" * 64), tmp_path / "models"
    with pytest.raises(UserError, match="checksum mismatch"):
        fetch.fetch_model("demo", models, directory)
    assert list(directory.iterdir()) == []


def test_failed_download_is_reported(tmp_path: Path) -> None:
    models, directory = served_model(tmp_path), tmp_path / "models"
    (tmp_path / "served.bin").unlink()
    with pytest.raises(UserError, match="download of model 'demo' failed"):
        fetch.fetch_model("demo", models, directory)
    assert list(directory.iterdir()) == []


def served_seedvr2_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sha256: str | None = None) -> Path:
    """Stand in for the pinned source archive, at a file:// address. Returns the archive."""
    top = f"ComfyUI-SeedVR2_VideoUpscaler-{catalog.SEEDVR2_COMMIT}"
    (tmp_path / "tree" / top).mkdir(parents=True)
    (tmp_path / "tree" / top / "LICENSE").write_text("Apache")
    archive = tmp_path / "repo" / "archive" / f"{catalog.SEEDVR2_COMMIT}.tar.gz"
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(tmp_path / "tree" / top, arcname=top)
    # fetch_seedvr2_code reads the pins from the fetch module's own names.
    monkeypatch.setattr(fetch, "SEEDVR2_REPO", (tmp_path / "repo").as_uri())
    monkeypatch.setattr(fetch, "SEEDVR2_ARCHIVE_SHA256", sha256 or fetch.sha256_of(archive))
    return archive


def test_seedvr2_code_is_unpacked_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive, directory = served_seedvr2_code(tmp_path, monkeypatch), tmp_path / "models"
    code = fetch.fetch_seedvr2_code(directory)
    assert (code / "LICENSE").read_text() == "Apache"
    assert list(directory.iterdir()) == [code]
    archive.unlink()
    assert fetch.fetch_seedvr2_code(directory) == code


def test_seedvr2_code_with_wrong_checksum_is_discarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    served_seedvr2_code(tmp_path, monkeypatch, sha256="0" * 64)
    with pytest.raises(UserError, match="checksum mismatch for SeedVR2 code"):
        fetch.fetch_seedvr2_code(tmp_path / "models")
    assert list((tmp_path / "models").iterdir()) == []


@pytest.mark.parametrize("name", ["bad.pth", "bad.safetensors", "bad.txt"])
def test_unloadable_model_file_is_reported(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_bytes(b"garbage")
    with pytest.raises(UserError, match="cannot load model file"):
        upscalers.XpuUpscaler(tmp_path / name, torch.device("cpu"))


# ---------------------------------------------------------------- command line


def test_user_errors_exit_with_1_and_one_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", str(tmp_path / "missing.png")]) == 1
    assert capsys.readouterr().err == f"arc-upscale: error: input does not exist: {tmp_path / 'missing.png'}\n"


def test_unknown_model_is_refused_by_fetch(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["fetch", "nope"]) == 1
    assert capsys.readouterr().err == "arc-upscale: error: unknown model: nope (use seedvr2, pbrify, fast)\n"


def test_too_small_tile_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--tile", "32", str(tmp_path)]) == 1
    assert "--tile must be larger than 32" in capsys.readouterr().err


# ---------------------------------------------------------------- project directory


def test_models_are_kept_under_the_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(catalog.ROOT_VARIABLE, str(tmp_path))
    assert catalog.models_dir() == tmp_path / "models"


def test_unset_project_root_is_reported(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv(catalog.ROOT_VARIABLE)
    assert cli.main(["fetch", "fast"]) == 1
    assert capsys.readouterr().err == (
        "arc-upscale: error: ARC_UPSCALE_ROOT is not set; run through the arc-upscale command, or make test\n"
    )
