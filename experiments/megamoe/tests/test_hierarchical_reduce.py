# Copyright (c) 2026, Lu Lu
"""Numerical/readiness checks using production reduction and unpermute code."""

from pathlib import Path
import shutil
import subprocess

import pytest


def function_source(text, name):
    start = text.index("__aicore__ inline void " + name + "(")
    opening = text.index("{", start)
    end, depth = opening + 1, 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def test_production_weighted_gateway_reduction_and_unpermute(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Production reduction checks require a C++ compiler")
    directory = Path(__file__).parent
    kernel = directory.parent / "csrc/kernels/mega_moe"
    dispatch = (kernel / "stage/mega_moe_hierarchical_dispatch.hpp").read_text(encoding="utf-8")
    fixture = (directory / "hierarchical_reduce_test.cpp").read_text(encoding="utf-8")
    source = tmp_path / "hierarchical_reduce.cpp"
    source.write_text(
        fixture.replace("/* PRODUCTION_WEIGHT_PACKER */", function_source(dispatch, "PackHierarchicalCombineWeights")),
        encoding="utf-8",
    )
    binary = tmp_path / "hierarchical_reduce"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-ffp-contract=off",
            f"-I{directory}",
            f"-I{directory / 'layout_stubs'}",
            f"-I{directory / 'combine_vf_stubs'}",
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
