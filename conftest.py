import pytest


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
