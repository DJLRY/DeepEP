# Copyright (c) 2026, Lu Lu
"""Concurrent control-flow checks for the production progress-preserving barriers."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_producers_keep_progress_without_crossing_barriers(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Combine synchronization checks require a C++ compiler")
    directory = Path(__file__).parent
    binary = tmp_path / "combine_sync"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(directory / "hierarchical_combine_sync_test.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
