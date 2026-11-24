"""A job queue in a spool directory, shared by the workers on one host.

A job is one JSON file. The directory it is in is its state:

    pending/          waiting, oldest first
    working/gpu<N>/   claimed by the worker on GPU N
    done/  failed/    finished; a failed job holds its error

Every move is a rename on one filesystem, which the kernel does atomically.
Of two workers that claim the same job, exactly one gets it. That needs a
local filesystem: not every network filesystem renames atomically.
"""

from __future__ import annotations

import fcntl
import os
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from arc_upscale.catalog import project_root
from arc_upscale.errors import UserError


class Job(BaseModel):
    model_config = ConfigDict(frozen=True)

    input: Path  # absolute, so that a worker in any directory finds it
    output: Path
    model: str  # a name from the catalog, or an absolute model file path
    tile: int | None  # None: the model's own default
    tiling: bool
    error: str | None = None  # set when the job failed


class Claimed(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    job: Job


def default_queue() -> Path:
    """`queue/` in the project, so every command on the host shares one queue."""
    return project_root() / "queue"


def load(path: Path) -> Job:
    try:
        return Job.model_validate_json(path.read_text())
    except ValidationError as error:
        raise UserError(f"invalid job file {path}: {error}") from error


def move(source: Path, target: Path) -> bool:
    """Rename `source` to `target`. False if `source` is gone: another worker moved it first."""
    try:
        source.rename(target)
    except FileNotFoundError:
        if source.exists() or not target.parent.is_dir():
            raise
        return False
    return True


class Spool:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.pending = root / "pending"
        self.done = root / "done"
        self.failed = root / "failed"
        # Files are written here first, then renamed into place, so nobody reads half a job.
        self.scratch = root / "scratch"
        for directory in (self.pending, self.done, self.failed, self.scratch, root / "working"):
            directory.mkdir(parents=True, exist_ok=True)

    def working(self, gpu: int) -> Path:
        directory = self.root / "working" / f"gpu{gpu}"
        directory.mkdir(exist_ok=True)
        return directory

    def write(self, job: Job, target: Path) -> None:
        scratch = self.scratch / target.name
        scratch.write_text(job.model_dump_json(indent=2) + "\n")
        scratch.replace(target)

    def submit(self, job: Job) -> Path:
        # Names sort by submission time. The process ID and random part keep them unique.
        target = self.pending / f"{time.time_ns():020d}-{os.getpid()}-{secrets.token_hex(4)}.json"
        self.write(job, target)
        return target

    def read(self, path: Path) -> Job:
        """Read a claimed job. An invalid file is moved to failed/, so that it is not claimed again."""
        try:
            return load(path)
        except UserError as error:
            target = self.failed / path.name
            path.rename(target)
            raise UserError(f"{error}; moved to {target}") from error

    def claim(self, gpu: int) -> Claimed | None:
        """Take the oldest pending job for the worker on `gpu`. None if there is none."""
        working = self.working(gpu)
        for source in sorted(self.pending.glob("*.json")):
            target = working / source.name
            if move(source, target):
                return Claimed(path=target, job=self.read(target))
        return None

    def finish(self, claimed: Claimed) -> Path:
        target = self.done / claimed.path.name
        claimed.path.rename(target)
        return target

    def fail(self, claimed: Claimed, error: str) -> Path:
        self.write(claimed.job.model_copy(update={"error": error}), claimed.path)
        target = self.failed / claimed.path.name
        claimed.path.rename(target)
        return target

    def requeue(self, gpu: int) -> list[Path]:
        """Put the jobs a stopped worker on `gpu` left behind back into pending/."""
        moved = []
        for source in sorted(self.working(gpu).glob("*.json")):
            target = self.pending / source.name
            source.rename(target)
            moved.append(target)
        return moved

    @contextmanager
    def worker_lock(self, gpu: int) -> Iterator[None]:
        """Held while a worker runs on `gpu`. A second worker on the same GPU is refused."""
        with (self.root / "working" / f"gpu{gpu}.lock").open("w") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise UserError(f"a worker already runs on GPU {gpu} for the queue {self.root}") from error
            yield

    def jobs(self, directory: Path) -> list[Claimed]:
        return [Claimed(path=path, job=load(path)) for path in sorted(directory.glob("*.json"))]

    def workers(self) -> list[Path]:
        return sorted(path for path in (self.root / "working").glob("gpu*") if path.is_dir())
