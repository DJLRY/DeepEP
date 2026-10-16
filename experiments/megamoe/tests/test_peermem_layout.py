# Copyright (c) 2026, Lu Lu

"""Compile the production layout and its budget against CANN arithmetic doubles."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_peermem_size_address_and_legacy_compatibility(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Production layout compilation requires g++ or clang++")
    source = Path(__file__).with_name("peermem_layout_test.cpp")
    stubs = Path(__file__).with_name("layout_stubs")
    binary = tmp_path / "peermem_layout_test"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{stubs}",
            str(source),
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
