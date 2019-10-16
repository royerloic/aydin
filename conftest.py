import pytest


@pytest.fixture(autouse=True)
def run_around_tests():
    # Code that will run before your test, for example:
    import os

    # os.environ["PYOPENCL_CTX"] = "0:2"

    # A test function will be run at this point
    yield
    # Code that will run after your test, for example:


def pytest_addoption(parser):
    parser.addoption(
        "--runheavy", action="store_true", default=False, help="run heavy tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "heavy: mark test as heavy to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runheavy"):
        # --runheavy given in cli: do not skip slow tests
        return
    skip_heavy = pytest.mark.skip(reason="need --runheavy option to run")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)
