# Copyright (c) 2026, Lu Lu

"""Exercise trace orchestration without CANN, including ranks without timers."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from deep_ep_experimental.megamoe import api, benchmark, validation
from deep_ep_experimental.megamoe.timer import breakdown


@pytest.mark.parametrize("rank", [0, 1, 16])
@pytest.mark.parametrize("warmup,iters", [(2, 1), (0, 5)])
def test_trace_only_runs_one_warmup_and_batch(tmp_path, monkeypatch, rank, warmup, iters):
    output = tmp_path / "output"
    config = dict(
        schema_version=1,
        model=dict(hidden=32, intermediate_hidden=32, num_experts=32, num_topk=1, num_shared_experts=0),
        bs_values=[2],
        warmup=warmup,
        iters=iters,
        seed=0,
        alignment=dict(rows=2, cols=2, repeats=1),
    )
    config_path = tmp_path / "case.json"
    config_path.write_text(json.dumps(config))
    (tmp_path / "_build_info.json").write_text("{}")
    monkeypatch.setattr(benchmark, "files", lambda _: tmp_path)
    monkeypatch.setenv("LOCAL_RANK", str(rank % 16))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "16")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--mode",
            "trace",
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
            "--input",
            "bf16",
            "--timer-ranks",
            "0,16",
        ],
    )
    torch = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())  # No profiler API available.
    group = SimpleNamespace(
        rank=lambda: rank,
        size=lambda: 32,
        all_gather=lambda value, stage: [value],
        barrier=MagicMock(),
        close=MagicMock(),
        abort=MagicMock(),
    )
    monkeypatch.setattr(api.StoreGroup, "from_env", lambda: group)

    def check(group, stage, error, signature=None):
        assert error is None, f"{stage}: {error}"

    monkeypatch.setattr(validation, "collective_check", check)
    buffer = MagicMock(timer_enabled=True, workspace_bytes=64, symmetric_buffer_bytes=128, num_max_tokens_per_rank=2)
    raw = np.array([11, 22, 33, 44], dtype="<i8")
    timer = buffer.allocate_timer.return_value
    timer.cpu.return_value.numpy.return_value = raw
    monkeypatch.setattr(api, "SymmBuffer", lambda *a, **kw: buffer)
    monkeypatch.setattr(api, "transform_weights_for_mega_moe", lambda *a, **kw: (object(), object()))
    calls = []
    monkeypatch.setattr(api, "fp8_fp4_mega_moe", lambda *a, **kw: calls.append(kw["timer"]))
    monkeypatch.setattr(benchmark, "collect_profile", lambda *a: pytest.fail("trace started performance profiler"))
    monkeypatch.setattr(benchmark, "accuracy", lambda *a, **kw: pytest.fail("trace started accuracy check"))
    monkeypatch.setattr(breakdown, "compact_events", lambda *a: [])
    monkeypatch.setattr(breakdown, "rank_breakdown", lambda events, rank: dict(rank=rank))
    monkeypatch.setattr(breakdown, "write_breakdown", lambda *a: None)

    # --mode trace must enable timers itself; no --timer flag is passed.
    benchmark.main()

    selected = rank in (0, 16)
    assert len(calls) == warmup + iters
    assert all(value is None for value in calls[:-1])
    assert calls[-1] is (timer if selected else None)
    assert buffer.allocate_timer.call_count == int(selected)
    timer_path = output / "bs_2" / "timer" / f"rank_{rank}_npu_time.bin"
    assert timer_path.exists() is selected
    if selected:
        assert timer_path.read_bytes() == raw.tobytes()
    for name in ("profiling.json", "kernel_samples.json", "profiler", "accuracy.json"):
        assert not list(output.rglob(name))
    manifest = json.loads((output / f"rank_{rank}" / "manifest.json").read_text())
    assert manifest["mode"] == "trace"
    buffer.destroy.assert_called_once()
    group.close.assert_called_once()
    group.abort.assert_not_called()
