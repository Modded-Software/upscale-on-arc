"""The job queue and the worker. No GPU: the model is replaced by nearest-neighbour scaling."""

import multiprocessing
import threading
from pathlib import Path

import pytest
import torch
from PIL import Image
from test_unit import NearestUpscaler

from arc_upscale import cli, gpus, jobs
from arc_upscale.errors import UserError
from arc_upscale.upscalers import Upscaler


def a_job(tmp_path: Path, name: str = "wall.png") -> jobs.Job:
    return jobs.Job(input=tmp_path / name, output=tmp_path / "out", model="fast", tile=None, tiling=False)


def names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob("*.json"))


# ---------------------------------------------------------------- the spool


def test_job_moves_from_pending_to_done(tmp_path: Path) -> None:
    spool = jobs.Spool(tmp_path / "queue")
    submitted = spool.submit(a_job(tmp_path))
    assert names(spool.pending) == [submitted.name]
    claimed = spool.claim(gpu=1)
    assert claimed is not None
    assert claimed.path == spool.root / "working" / "gpu1" / submitted.name
    assert claimed.job == a_job(tmp_path)
    assert names(spool.pending) == []
    assert spool.finish(claimed) == spool.done / submitted.name
    assert names(spool.working(1)) == []
    assert names(spool.done) == [submitted.name]


def test_failed_job_keeps_its_error(tmp_path: Path) -> None:
    spool = jobs.Spool(tmp_path / "queue")
    spool.submit(a_job(tmp_path))
    claimed = spool.claim(gpu=0)
    assert claimed is not None
    target = spool.fail(claimed, "the GPU ran out of memory")
    assert jobs.load(target).error == "the GPU ran out of memory"
    assert names(spool.working(0)) == []


def test_jobs_are_claimed_oldest_first(tmp_path: Path) -> None:
    spool = jobs.Spool(tmp_path / "queue")
    submitted = [spool.submit(a_job(tmp_path, f"{index}.png")) for index in range(5)]
    claimed = [spool.claim(gpu=0) for _ in submitted]
    assert [entry.path.name for entry in claimed if entry is not None] == [path.name for path in submitted]
    assert spool.claim(gpu=0) is None


WORKERS = 4


def claim_all(root: Path, gpu: int, start: threading.Barrier) -> list[str]:
    """Claim and finish jobs until none are left. Runs in a process of its own."""
    spool = jobs.Spool(root)
    taken = []
    start.wait(timeout=120)  # all workers start together, so that they race
    while (claimed := spool.claim(gpu)) is not None:
        taken.append(claimed.path.name)
        spool.finish(claimed)
    return taken


def test_concurrent_workers_never_take_the_same_job(tmp_path: Path) -> None:
    spool = jobs.Spool(tmp_path / "queue")
    submitted = sorted(spool.submit(a_job(tmp_path, f"{index}.png")).name for index in range(300))
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager, context.Pool(WORKERS) as pool:
        start = manager.Barrier(WORKERS)
        taken = pool.starmap(claim_all, [(spool.root, gpu, start) for gpu in range(WORKERS)])
    every = [name for worker in taken for name in worker]
    assert sorted(every) == submitted  # each job exactly once
    assert names(spool.done) == submitted
    assert sum(1 for worker in taken if worker) > 1, "only one worker got jobs: the race was not exercised"


def test_jobs_of_a_stopped_worker_are_requeued(tmp_path: Path) -> None:
    spool = jobs.Spool(tmp_path / "queue")
    spool.submit(a_job(tmp_path))
    claimed = spool.claim(gpu=0)
    assert claimed is not None
    assert spool.requeue(gpu=1) == []
    assert spool.requeue(gpu=0) == [spool.pending / claimed.path.name]
    assert names(spool.working(0)) == []


def test_one_worker_per_gpu(tmp_path: Path) -> None:
    spool = jobs.Spool(tmp_path / "queue")
    with spool.worker_lock(0):
        with pytest.raises(UserError, match="a worker already runs on GPU 0"), spool.worker_lock(0):
            pass
        with spool.worker_lock(1):
            pass
    with spool.worker_lock(0):
        pass


def test_invalid_job_file_is_moved_to_failed(tmp_path: Path) -> None:
    spool = jobs.Spool(tmp_path / "queue")
    (spool.pending / "1-broken.json").write_text("{not json")
    with pytest.raises(UserError, match="invalid job file"):
        spool.claim(gpu=0)
    assert names(spool.failed) == ["1-broken.json"]
    assert spool.claim(gpu=0) is None


# ---------------------------------------------------------------- the commands


@pytest.fixture
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the GPU and the model. Returns the models that were loaded, in order."""
    loaded: list[str] = []

    def load(model: str, _device: torch.device) -> tuple[Upscaler, str]:
        loaded.append(model)
        return NearestUpscaler(), "nearest"

    monkeypatch.setattr(cli, "pick_gpu", lambda index: gpus.Gpu(index=index, name="test GPU", memory=0, pci=None))
    monkeypatch.setattr(cli, "load_upscaler", load)
    return loaded


def test_submit_stores_absolute_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    Image.new("RGB", (8, 8)).save("wall.png")
    assert cli.main(["submit", "--queue", "q", "-m", "fast", "-o", "big", "wall.png"]) == 0
    [claimed] = jobs.Spool(tmp_path / "q").jobs(tmp_path / "q" / "pending")
    assert claimed.job == jobs.Job(
        input=tmp_path / "wall.png", output=tmp_path / "big", model="fast", tile=None, tiling=False
    )


def test_submit_refuses_bad_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    queue = str(tmp_path / "q")
    assert cli.main(["submit", "--queue", queue, str(tmp_path / "missing.png")]) == 1
    assert "input does not exist" in capsys.readouterr().err
    assert cli.main(["submit", "--queue", queue, "-m", str(tmp_path / "none.pth"), str(tmp_path)]) == 1
    assert "model not found" in capsys.readouterr().err
    assert cli.main(["submit", "--queue", queue, "--tile", "8", str(tmp_path)]) == 1
    assert names(tmp_path / "q" / "pending") == []


def test_worker_runs_jobs_and_records_failures(
    tmp_path: Path, no_gpu: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    queue = str(tmp_path / "q")
    Image.new("RGB", (8, 6), (10, 20, 30)).save(tmp_path / "wall.png")
    (tmp_path / "broken.png").write_bytes(b"not a png")
    for name in ("wall.png", "broken.png", "wall.png"):
        out = str(tmp_path / f"out-{name}")
        assert cli.main(["submit", "--queue", queue, "-m", "fast", "-o", out, str(tmp_path / name)]) == 0
    assert cli.main(["worker", "--queue", queue, "-g", "1", "--until-empty"]) == 0

    with Image.open(tmp_path / "out-wall.png" / "wall_4x.png") as image:
        assert image.size == (32, 24)
    spool = jobs.Spool(tmp_path / "q")
    assert len(names(spool.done)) == 2
    [failed] = spool.jobs(spool.failed)
    assert failed.job.error == f"not a readable image: {tmp_path / 'broken.png'}"
    assert "arc-upscale: job failed: not a readable image" in capsys.readouterr().err
    assert no_gpu == ["fast"], "the model is loaded once, not per job"


def test_worker_requeues_its_own_leftovers(tmp_path: Path, no_gpu: list[str]) -> None:
    Image.new("RGB", (8, 8)).save(tmp_path / "wall.png")
    spool = jobs.Spool(tmp_path / "q")
    spool.submit(a_job(tmp_path))
    assert spool.claim(gpu=0) is not None  # as if a worker on GPU 0 had crashed
    assert cli.main(["worker", "--queue", str(spool.root), "-g", "1", "--until-empty"]) == 0
    assert names(spool.working(0)) != [], "a worker leaves other GPUs' jobs alone"
    assert cli.main(["worker", "--queue", str(spool.root), "-g", "0", "--until-empty"]) == 0
    assert names(spool.working(0)) == []
    assert len(names(spool.done)) == 1


def test_model_slot_holds_one_model(no_gpu: list[str]) -> None:
    slot = cli.ModelSlot(torch.device("cpu"))
    first = slot.get("fast")
    assert slot.get("fast") is first
    assert slot.get("pbrify") is not first
    assert no_gpu == ["fast", "pbrify"]


def test_queue_command_shows_the_state(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spool = jobs.Spool(tmp_path / "q")
    for index in range(3):
        spool.submit(a_job(tmp_path, f"{index}.png"))
    running = spool.claim(gpu=1)
    broken = spool.claim(gpu=0)
    assert running is not None and broken is not None
    spool.fail(broken, "the GPU ran out of memory")
    assert cli.main(["queue", "--queue", str(spool.root)]) == 0
    out = capsys.readouterr().out
    assert f"queue {spool.root}: 1 pending, 0 done, 1 failed" in out
    assert f"gpu1: {running.path.name}  {tmp_path / '0.png'}" in out
    assert f"failed: {broken.path.name}  {tmp_path / '1.png'}: the GPU ran out of memory" in out


def test_default_queue_is_in_the_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARC_UPSCALE_ROOT", str(tmp_path))
    assert jobs.default_queue() == tmp_path / "queue"
