#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""
This is a configuration file for pytest containing customizations and fixtures.

In VSCode, Code Coverage is recorded in config.xml. Delete this file to reset reporting.
"""

from __future__ import annotations

# Force JAX to expose 4 CPU devices so that multi-device tests (which simulate
# multiple GPUs using CPU devices) can run without real GPU hardware.  This
# must be set before any JAX import so it takes effect during initialisation.
import os
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=4')

from typing import List

import pytest
from _pytest.nodes import Item


def pytest_collection_modifyitems(items: list[Item]):
    for item in items:
        if "_int_" in item.nodeid:
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def unit_test_mocks(monkeypatch: None):
    """Include Mocks here to execute all commands offline and fast."""
    pass
