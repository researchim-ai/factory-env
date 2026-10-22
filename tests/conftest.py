"""Shared fixtures."""

from __future__ import annotations

import pytest

from factoryrl.scenarios.builtins import (
    choice_instance,
    demo_instance,
    edd_instance,
    flow_demo_instance,
)


@pytest.fixture
def demo():
    return demo_instance()


@pytest.fixture
def choice():
    return choice_instance()


@pytest.fixture
def edd():
    return edd_instance()


@pytest.fixture
def flow_demo():
    return flow_demo_instance()
