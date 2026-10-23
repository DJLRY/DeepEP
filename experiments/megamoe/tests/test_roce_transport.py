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
    compile_and_run(tmp_path, backend, "roce_transport_test.cpp")


@pytest.mark.parametrize("backend", ["XSCALE", "HNS_1825"])
def test_hierarchical_dispatch_full_batch_and_completion_contract(tmp_path, backend):
    compile_and_run(
        tmp_path,
        backend,
        "hierarchical_dispatch_test.cpp",
        stub_directories=("dispatch_stubs", "roce_stubs", "layout_stubs"),
    )


@pytest.mark.parametrize("backend", ["XSCALE", "HNS_1825"])
@pytest.mark.parametrize("timer", [False, True], ids=["timer-off", "timer-on"])
def test_rdma_timer_preserves_compute_and_transport(tmp_path, backend, timer):
    compile_and_run(tmp_path, backend, "rdma_timer_test.cpp", timer)


@pytest.mark.parametrize("timer", [False, True], ids=["timer-off", "timer-on"])
def test_startup_timer_closes_at_first_work_without_extra_barriers(tmp_path, timer):
    compile_and_run(tmp_path, None, "startup_timer_test.cpp", timer)


@pytest.mark.parametrize("timer", [False, True], ids=["timer-off", "timer-on"])
def test_combine_progress_timer_preserves_state_and_excludes_paused_work(tmp_path, timer):
    compile_and_run(tmp_path, None, "combine_progress_timer_test.cpp", timer)


def compile_and_run(tmp_path, backend, source_name, timer=False, stub_directories=("roce_stubs",)):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Transport helper compilation requires g++ or clang++")
    source = Path(__file__).with_name(source_name)
    fixture_directory = source.parent
    if source_name == "hierarchical_dispatch_test.cpp":
        production = (fixture_directory.parent / "csrc/kernels/mega_moe/stage/mega_moe_send_mask.hpp").read_text(
            encoding="utf-8"
        )
        start = production.index("__aicore__ inline void CalcRailRelayExpertRange(")
        opening = production.index("{", start)
        end, depth = opening + 1, 1
        while depth:
            depth += (production[end] == "{") - (production[end] == "}")
            end += 1
        generated = tmp_path / source_name
        generated.write_text(
            source.read_text(encoding="utf-8").replace("/* PRODUCTION_RELAY_RANGE */", production[start:end]),
            encoding="utf-8",
        )
        source = generated
    binary = tmp_path / "roce_transport_test"
    command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror"]
    if source_name == "hierarchical_dispatch_test.cpp":
        # The fixture checks full symmetric heaps, both arrival orders and all
        # relay partitions. Optimize its byte comparisons, retaining assertions.
        command.append("-O2")
    if backend is not None:
        command.append(f"-DACLSHMEMI_RDMA_K_BACKEND_{backend}=1")
    if timer:
        command.append("-DDEEPEP_MEGAMOE_TIMER=1")
    command += [f"-I{Path(__file__).with_name(directory)}" for directory in stub_directories]
    command += [f"-I{fixture_directory}", str(source), "-o", str(binary)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        timeout=30 if source_name == "hierarchical_dispatch_test.cpp" else 10,
    )
    assert result.returncode == 0, result.stderr
