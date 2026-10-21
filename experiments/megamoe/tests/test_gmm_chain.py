# Copyright (c) 2026, Lu Lu

"""CPU checks for benchmark inputs, actual tensor chaining, and timing boundaries."""

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "bench_gmm_chain", ROOT / "experiments/megamoe/scripts/bench_gmm_chain.py"
)
chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain)


@pytest.mark.parametrize("rank", [0, 14, 16, 30, 31])
def test_default_routing_reproduces_actual_dispatch_rows(rank):
    args = chain.arguments(["--rank", str(rank)])
    config = chain.case_config(args)
    x, counts, coords = chain.routed_input(config, 32, rank, 32, "balanced")
    assert counts.tolist() == [22] * 4 + [21] * 8
    assert x.shape == (256, 3584)
    expert_ids = []
    for source in range(32):
        original, ids, _ = chain.inputs(config, source, 32, "balanced", world=32)
        rows = np.flatnonzero(coords[:, 0] == source)
        token, slot = coords[rows, 1], coords[rows, 2]
        np.testing.assert_array_equal(x[rows], original[token])
        expert_ids.extend(zip(rows, ids[token, slot]))
    assert [expert for _, expert in sorted(expert_ids)] == np.repeat(
        np.arange(rank * 12, (rank + 1) * 12), counts
    ).tolist()
    assert len(set(map(tuple, coords))) == 256
    plan = chain.describe(args, counts)
    assert plan["gmm1"] == dict(m=256, k=3584, n=2560)
    assert plan["gmm2"] == dict(m=256, k=1280, n=3584)


@pytest.mark.parametrize("routing", ["random", "skewed"])
def test_sparse_routes_preserve_empty_experts_and_all_assignments(routing):
    args = chain.arguments(["--ep-size", "2", "--experts", "24", "--bs", "2", "--topk", "1", "--hidden", "64"])
    config = chain.case_config(args)
    total = 0
    for rank in range(2):
        x, counts, coords = chain.routed_input(config, 2, rank, 2, routing)
        expected = []
        for source in range(2):
            _, ids, _ = chain.inputs(config, source, 2, routing, world=2)
            for token in range(2):
                if ids[token, 0] // 12 == rank:
                    expected.append((int(ids[token, 0]), source, token, 0))
        assert coords.tolist() == [list(row[1:]) for row in sorted(expected)]
        assert counts.sum() == len(x) == len(expected)
        assert len(counts) == 12 and np.any(counts == 0)
        total += len(x)
    assert total == 4


def test_dry_run_without_device_imports(capsys):
    chain.main(["--dry-run"])
    assert json.loads(capsys.readouterr().out)["plan"]["routed_rows"] == 256


def test_failed_device_run_keeps_manifest_in_fresh_directory(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("Device API unavailable")

    monkeypatch.setattr(chain, "run", fail)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Device API unavailable"):
            chain.main(["--output", str(tmp_path), "--hidden", "64"])
    reports = list(tmp_path.glob("rank_0_*/results.json"))
    assert len(reports) == 2
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["status"] == "failed" and "Device API unavailable" in report["error"]
        assert report["plan"]["routed_rows"] == 256


def test_operator_chain_uses_real_fused_outputs_and_prepares_only_once():
    events = []

    class Tensor:
        def __init__(self, data, dtype="uint8"):
            self.data, self.dtype = data, dtype

        def view(self, dtype):
            return Tensor(self.data, dtype)

        def to(self, device):
            events.append(("copy", device))
            return self

        def transpose(self, a, b):
            return Tensor(np.swapaxes(self.data, a, b), self.dtype)

    torch = SimpleNamespace(from_numpy=Tensor, float8_e4m3fn="fp8", float32="fp32", bfloat16="bf16")
    activation, scale, result = object(), object(), object()

    def format_cast(weight, fmt, **kw):
        events.append(("format", fmt))
        assert fmt == 29 and kw == dict(customize_dtype="fp8", input_dtype="fp4")
        return weight

    def fused(x, weights, scales, x_scale, groups, **kw):
        events.append(("fused",))
        assert x.dtype == "fp8" and x_scale.data.shape == (3, 1, 2)
        assert groups.data.tolist() == [2, 0, 1]
        assert kw["group_list_type"] == 1 and kw["quant_dtype"] == "fp8"
        assert kw["weight_scale_dtype"] == kw["x_scale_dtype"] == "e8m0"
        np.testing.assert_array_equal(weights[0].data, raw[0][0].swapaxes(-1, -2))
        np.testing.assert_array_equal(scales[0].data, raw[0][1].swapaxes(-3, -2))
        return activation, scale

    def gmm2(x, weights, **kw):
        events.append(("gmm2",))
        assert x[0] is activation and kw["per_token_scale"][0] is scale
        assert "scale" not in kw and kw["output_dtype"] == "bf16"
        assert kw["weight_dtype"] == "fp4" and kw["per_token_scale_dtype"] == "e8m0"
        assert kw["group_list_type"] == 1 and kw["group_list"].data.tolist() == [2, 0, 1]
        np.testing.assert_array_equal(weights[0].data, raw[1][0].swapaxes(-1, -2))
        np.testing.assert_array_equal(kw["antiquant_scale"][0].data, raw[1][1].swapaxes(-3, -2))
        return [result]

    npu = SimpleNamespace(
        npu_format_cast=format_cast,
        npu_grouped_matmul_swiglu_quant_v2=fused,
        npu_grouped_matmul=gmm2,
        float4_e2m1fn_x2="fp4",
        float8_e8m0fnu="e8m0",
    )
    args = chain.arguments(["--hidden", "64", "--intermediate", "128"])
    raw = chain.make_weights(3, 64, 128, 23)
    call, _ = chain.prepare_chain(torch, npu, args, np.ones((3, 64), dtype=np.float32), np.array([2, 0, 1]), raw)
    assert [entry for entry in events if entry[0] == "format"] == [("format", 29)] * 2
    events.clear()
    assert call() is result and call() is result
    assert events == [("fused",), ("gmm2",)] * 2


def test_reference_handles_empty_experts_and_detects_corruption():
    raw = chain.make_weights(3, 64, 128, 23)
    counts = np.array([2, 0, 1])
    quantized = chain.quantize_input(np.random.default_rng(4).standard_normal((3, 64)).astype(np.float32))
    x = chain.dequantize_input(*quantized)
    expected = np.concatenate(
        [
            chain.expert_forward(x[start:stop], *(chain.unpack_weight((w[e], s[e])) for w, s in raw))
            for e, start, stop in [(0, 0, 2), (2, 2, 3)]
        ]
    )
    report = chain.validate_output(expected, quantized, counts, raw, 2)
    assert report["passed"] and report["checked_rows"] == 3 and report["relative_rmse"] == 0
    assert not chain.validate_output(expected + 1, quantized, counts, raw, 2)["passed"]
    assert chain.validate_output(expected, quantized, counts, raw, 0)["passed"] is None
    expected[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        chain.validate_output(expected, quantized, counts, raw, 0)


def capture_rows():
    return [
        dict(Name=name, **{"Start Time(us)": start, "Duration(us)": duration, "Device_id": "0", "Stream ID": "4"})
        for name, start, duration in [
            ("GroupedMatmulSwigluQuantV2", 100, 10),
            ("GroupedMatmul", 112, 5),
            ("GroupedMatmulSwigluQuantV2", 130, 7),
            ("GroupedMatmul", 138, 4),
        ]
    ]


def save_capture(tmp_path, rows):
    with (tmp_path / "kernel_details.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(capture_rows()[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_capture_distinguishes_kernel_sum_from_chain_span(tmp_path):
    save_capture(tmp_path, list(reversed(capture_rows())))
    pairs = chain.read_pairs(tmp_path, 2)
    assert pairs[0]["kernel_sum_us"] == 15 and pairs[0]["chain_span_us"] == 17
    report = chain.summarize(pairs)
    assert report["rounds"] == 1 and report["first_selected_iteration"] == 1
    assert report["kernel_sum_us"]["mean"] == 11 and report["chain_span_us"]["mean"] == 12
    assert chain.summarize(pairs + [pairs[0]])["rounds"] == 2


@pytest.mark.parametrize(
    "index,key,value,error",
    [
        (0, "Name", "Copy", "Unexpected kernel"),
        (0, "Name", "GroupedMatmul", "ordering"),
        (0, "Duration(us)", "nan", "Non-finite"),
        (0, "Duration(us)", 0, "Non-finite"),
        (0, "Start Time(us)", "bad", "Invalid kernel timing"),
        (1, "Start Time(us)", 100, "Duplicate"),
        (1, "Start Time(us)", 105, "Overlapping"),
        (2, "Start Time(us)", 115, "Overlapping"),
        (3, "Device_id", "1", "devices/streams"),
        (3, "Stream ID", "8", "devices/streams"),
        (3, "Stream ID", "", "Missing device/stream"),
    ],
)
def test_capture_rejects_ambiguous_or_malformed_samples(tmp_path, index, key, value, error):
    rows = capture_rows()
    rows[index][key] = value
    save_capture(tmp_path, rows)
    with pytest.raises(ValueError, match=error):
        chain.read_pairs(tmp_path, 2)


def test_missing_capture_is_not_a_zero_timing(tmp_path):
    with pytest.raises(ValueError, match="found 0 kernels"):
        chain.read_pairs(tmp_path, 2)


def test_profiler_never_synchronizes_between_formal_calls(tmp_path, monkeypatch):
    events = []

    class Profile:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *args):
            events.append("exit")

        def step(self):
            events.append("step")

    profiler = SimpleNamespace(
        _ExperimentalConfig=lambda **kw: kw,
        ExportType=SimpleNamespace(Text=0),
        ProfilerLevel=SimpleNamespace(Level1=1),
        ProfilerActivity=SimpleNamespace(CPU=0, NPU=1),
        profile=lambda **kw: Profile(),
        schedule=lambda **kw: kw,
        tensorboard_trace_handler=lambda path: None,
    )
    torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: events.append("sync")))
    monkeypatch.setattr(chain, "read_pairs", lambda directory, iterations: (directory, iterations))
    assert chain.collect_profile(
        lambda: events.append("call"), torch, SimpleNamespace(profiler=profiler), 3, tmp_path
    ) == (tmp_path, 3)
    assert events == ["enter", "sync", "call", "call", "call", "sync", "step", "exit"]
