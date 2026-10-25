# Copyright (c) 2026, Lu Lu

"""Compile real pinned SHMEM WQE builders; emulate only device types and queue I/O."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def function(source, name, *, template=True):
    position = source.index(name + "(")
    start = source.rfind("\ntemplate <" if template else "\nACLSHMEM_DEVICE", 0, position) + 1
    opening = source.index("\n{", position) + 1
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def sdk_source(path):
    checkout = Path(os.environ.get("DEEPEP_SHMEM_TEST_SOURCE", ROOT / "third-party/shmem"))
    revision = json.loads((ROOT / "dependencies.lock.json").read_text())["shmem"]["revision"]
    result = subprocess.run(
        ["git", "-C", str(checkout), "show", f"{revision}:{path}"], capture_output=True, encoding="utf-8"
    )
    if result.returncode:
        pytest.skip("Pinned SHMEM Git sources required; initialize submodule or set DEEPEP_SHMEM_TEST_SOURCE")
    return result.stdout


def contract_source():
    common = sdk_source("src/device/shmemi_device_common.hpp")
    udma = sdk_source("src/device/gm2gm/engine/shmem_device_udma.hpp")
    names = (
        "aclshmemi_udma_check_config",
        "aclshmemi_udma_require_submit_cqe",
        "aclshmemi_udma_build_flag",
        "aclshmemi_fill_sge_header_ctx",
        "aclshmemi_udma_get_reduce_attr",
        "aclshmemi_udma_fill_sqe_base_ctx",
        "aclshmemi_udma_fill_sqe_ctx",
        "aclshmemi_udma_stage_send_wqe",
    )
    production = "\n".join(
        [
            function(common, "aclshmemi_debug_assert_remote_mr_addr", template=False),
            function(common, "aclshmemi_remote_mr_addr", template=False),
        ]
        + [function(udma, name) for name in names]
    )
    combine = (ROOT / "csrc/kernels/combine/combine.cpp").read_text(encoding="utf-8")
    batch = combine[combine.index("struct CrossPutBatch") :]
    start = batch.index("    __aicore__ inline void Add(")
    body = batch.index("\n    {", start) + 5
    depth, end = 1, body + 1
    while depth:
        depth += (batch[end] == "{") - (batch[end] == "}")
        end += 1
    fixture = (Path(__file__).with_name("shmem_udma_abi_test.cpp")).read_text(encoding="utf-8")
    return fixture.replace("/* PINNED_SHMEM_BUILDERS */", production).replace(
        "/* PRODUCTION_COMBINE_ADD */", batch[start:end]
    )


@pytest.mark.parametrize("debug", [False, True], ids=["release", "sdk-debug"])
def test_live_and_symmetric_wqe_addresses_with_pinned_sdk(tmp_path, debug):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    source = tmp_path / "contract.cpp"
    source.write_text(contract_source(), encoding="utf-8")
    binary = tmp_path / ("contract.exe" if os.name == "nt" else "contract")
    result = subprocess.run(
        [compiler, "-std=c++17", f"-DTEST_DEBUG={int(debug)}", "-iquote", str(ROOT), str(source), "-o", str(binary)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
