"""Downloads of model files and of the pinned SeedVR2 code, all checksum-verified."""

from __future__ import annotations

import contextlib
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from arc_upscale.catalog import (
    MODELS,
    SEEDVR2,
    SEEDVR2_ARCHIVE_SHA256,
    SEEDVR2_COMMIT,
    SEEDVR2_REPO,
    SEEDVR2_WEIGHTS,
    ModelFile,
    models_dir,
)
from arc_upscale.errors import UserError


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, sha256: str, path: Path, what: str) -> None:
    """Download a file to `path`.

    The download is verified before it is moved into place, so a bad file never
    ends up where a later run would use it.
    """
    print(f"downloading {what}: {path.name}", file=sys.stderr)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(url) as response, part.open("wb") as out:
            shutil.copyfileobj(response, out)
    except urllib.error.URLError as error:
        part.unlink(missing_ok=True)
        raise UserError(f"download of {what} failed: {error.reason}") from error
    if sha256_of(part) != sha256:
        part.unlink()
        raise UserError(f"checksum mismatch for {what}; download discarded")
    part.replace(path)


def fetch_model(name: str, models: dict[str, ModelFile] = MODELS, directory: Path | None = None) -> Path:
    """Return the path of a named model file, downloading it first if it is absent.

    `directory` is the project's models/ unless a test passes its own.
    """
    if directory is None:
        directory = models_dir()
    model = models[name]
    path = directory / model.file
    if not path.exists():
        download(model.url, model.sha256, path, f"model '{name}'")
    return path


def fetch_seedvr2_code(directory: Path | None = None) -> Path:
    """Return the directory of the pinned SeedVR2 source, downloading it first if it is absent."""
    if directory is None:
        directory = models_dir()
    code = directory / f"seedvr2-code-{SEEDVR2_COMMIT[:12]}"
    if code.exists():
        return code
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as scratch:
        archive = Path(scratch) / "seedvr2-code.tar.gz"
        download(f"{SEEDVR2_REPO}/archive/{SEEDVR2_COMMIT}.tar.gz", SEEDVR2_ARCHIVE_SHA256, archive, "SeedVR2 code")
        with tarfile.open(archive) as tar:
            tar.extractall(scratch, filter="data")
        (Path(scratch) / f"ComfyUI-SeedVR2_VideoUpscaler-{SEEDVR2_COMMIT}").replace(code)
    return code


def fetch_seedvr2(directory: Path | None = None) -> tuple[Path, Path]:
    """Return the SeedVR2 code and weights directories, downloading what is absent.

    The code directory is put on sys.path: upstream is not a package and imports as `src`.
    """
    if directory is None:
        directory = models_dir()
    code, weights = fetch_seedvr2_code(directory), directory / SEEDVR2
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    with contextlib.redirect_stdout(sys.stderr):  # upstream logs with print()
        from src.utils.debug import Debug
        from src.utils.downloads import download_weight

        if not download_weight(*SEEDVR2_WEIGHTS, model_dir=str(weights), debug=Debug(enabled=False)):
            raise UserError("download of the SeedVR2 weights failed")
    return code, weights
