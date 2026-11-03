# Copyright (c) 2026, Lu Lu
"""Deferred-DMA lifetime checks around the production synchronous single-row Combine function."""

from pathlib import Path
import re
import shutil
import subprocess

import pytest


def test_single_row_visibility_events_and_activation_reuse(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Single-row Combine checks require a C++ compiler")
    directory = Path(__file__).parent
    production = (directory.parent / "csrc/kernels/mega_moe/stage/mega_moe_gmm2_combine.hpp").read_text(
        encoding="utf-8"
    )
    routes = production[production.index("struct CombineTokenRoute") : production.index("// 为 tile 内")]
    single_row_code = production[
        production.index("struct WaveCombineScratch") : production.index("// 非 layered 路径统一使用的 Combine 行环")
    ]
    # The extracted range also declares ring-only constants; omit those because
    # this fixture compiles the standalone single-row specialization.
    single_row_code = re.sub(
        r"^constexpr uint32_t WAVE_COMBINE_(?!UB_BASE\b)[^;]+;\n", "", single_row_code, flags=re.MULTILINE
    )
    source = tmp_path / "single_row_combine.cpp"
    source.write_text(
        (directory / "single_row_combine_test.cpp")
        .read_text(encoding="utf-8")
        .replace("/* PRODUCTION_ROUTES */", routes)
        .replace("/* PRODUCTION_SINGLE_ROW */", single_row_code),
        encoding="utf-8",
    )
    binary = tmp_path / "single_row_combine"
    result = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
