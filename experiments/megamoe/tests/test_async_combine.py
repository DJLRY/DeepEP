# Copyright (c) 2026, Lu Lu
"""Deferred-DMA lifetime checks around the production async Combine functions."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_async_row_visibility_events_and_activation_overlap(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Async Combine checks require a C++ compiler")
    directory = Path(__file__).parent
    production = (directory.parent / "csrc/kernels/mega_moe/stage/mega_moe_gmm2_combine.hpp").read_text(
        encoding="utf-8"
    )
    routes = production[production.index("struct CombineTokenRoute") : production.index("// 为 tile 内")]
    async_code = production[
        production.index("struct WaveCombineScratch") : production.index("constexpr uint32_t WAVE_COMBINE_MIN")
    ]
    source = tmp_path / "async_combine.cpp"
    source.write_text(
        (directory / "async_combine_test.cpp")
        .read_text(encoding="utf-8")
        .replace("/* PRODUCTION_ROUTES */", routes)
        .replace("/* PRODUCTION_ASYNC */", async_code),
        encoding="utf-8",
    )
    binary = tmp_path / "async_combine"
    result = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
