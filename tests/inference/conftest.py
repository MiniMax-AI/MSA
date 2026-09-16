"""Shared inference test-suite selection."""

from __future__ import annotations

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("msa-inference")
    group.addoption(
        "--msa-inference-suite",
        choices=("smoke", "full"),
        default=os.environ.get("MINIMAX_INFERENCE_TEST_SUITE", "smoke"),
        help="run the 96-case smoke suite or 256-case full suite",
    )


def pytest_configure(config: pytest.Config) -> None:
    os.environ["MINIMAX_INFERENCE_TEST_SUITE"] = config.getoption(
        "--msa-inference-suite"
    )
