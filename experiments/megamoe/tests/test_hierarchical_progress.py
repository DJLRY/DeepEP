# Copyright (c) 2026, Lu Lu

"""Exercise the production C++ progress loop with instrumented device doubles.

The executable includes the real hierarchical-combine header. Its doubles only
replace DMA, flag loads, and asynchronous Put submission. These are CPU control
flow checks, not evidence of NPU cache coherence or NIC completion.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def hierarchical_progress_binary(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Hierarchical progress compilation requires g++ or clang++")
    directory = Path(__file__).parent
    binary = tmp_path_factory.mktemp("hierarchical-progress") / "hierarchical_progress_test"
    command = [
        compiler,
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(directory / "hierarchical_progress_test.cpp"),
        "-o",
        str(binary),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return binary


@pytest.mark.parametrize(
    "scenario",
    [
        "routing",
        "late-ready",
        "fairness",
        "budgets",
        "epochs",
        "completed",
        "metadata",
        "drain",
        "topology",
    ],
)
def test_hierarchical_progress_contract(hierarchical_progress_binary, scenario):
    result = subprocess.run(
        [str(hierarchical_progress_binary), scenario],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
