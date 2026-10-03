import pytest


def pytest_addoption(parser):
    parser.addoption('--run-experimental',action='store_true',default=False,
                     help='build and test isolated research kernels that are not used by deployment')


def pytest_configure(config):
    config.addinivalue_line('markers','experimental: isolated candidate kernels, not deployment defaults')


def pytest_collection_modifyitems(config,items):
    if config.getoption('--run-experimental'):return
    skip=pytest.mark.skip(reason='isolated research kernel: pass --run-experimental to build and test')
    for item in items:
        if 'experimental' in item.keywords:item.add_marker(skip)
