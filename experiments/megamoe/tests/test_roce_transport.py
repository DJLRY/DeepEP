# Copyright (c) 2026, Lu Lu

"""Compile the real transport helper against asynchronous SHMEM/event doubles.

These checks cover API use and protocol sequencing, not NPU/NIC visibility.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("backend", ["XSCALE", "HNS_1825", None], ids=["xscale", "hns-1825", "mte-only"])
def test_roce_transport_resource_and_completion_contract(tmp_path, backend):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Transport helper compilation requires g++ or clang++")
    source = Path(__file__).with_name("roce_transport_test.cpp")
    stubs = Path(__file__).with_name("roce_stubs")
    binary = tmp_path / "roce_transport_test"
    command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror"]
    if backend is not None:
        command.append(f"-DACLSHMEMI_RDMA_K_BACKEND_{backend}=1")
    command += [f"-I{stubs}", str(source), "-o", str(binary)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
