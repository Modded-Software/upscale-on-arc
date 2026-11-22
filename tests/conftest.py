import pytest

from arc_upscale import gpus


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--gpu", type=int, default=0, help="GPU index for the GPU tests (see: arc-upscale gpus)")


@pytest.fixture(scope="session")
def gpu(request: pytest.FixtureRequest) -> gpus.Gpu:
    return gpus.pick_gpu(request.config.getoption("--gpu"))
