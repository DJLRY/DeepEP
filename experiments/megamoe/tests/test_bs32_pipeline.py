# Copyright (c) 2026, Lu Lu
"""Keep independent experiment pairs and timer diagnostics consistent."""

from pathlib import Path
import json
import runpy

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/bs32_pipeline.py"
MODULE = runpy.run_path(str(SCRIPT))


def test_independent_cases_only_change_one_factor():
    cases = MODULE["CASES"]
    pairs = {
        "q0_single_row": ("q0_scan", {"--combine-single-row"}),
        "q8_double_buffer": ("q8_dispatch14", {"--reduce-double-buffer"}),
    }
    for prefix, baseline in (("q0", "q0_scan"), ("q8", "q8_double_buffer")):
        for suffix, switch in (("skip", "skip-empty"), ("lookahead", "lookahead"), ("row", "single-row")):
            pairs[f"{prefix}_dispatch_{suffix}"] = (baseline, {"--dispatch-" + switch})
    for name, (baseline, additions) in pairs.items():
        assert set(cases[name]) - set(cases[baseline]) == additions
        assert set(cases[baseline]) - set(cases[name]) == set()
    assert len(MODULE["INDEPENDENT_CASES"]) == 8
    assert not any(name.endswith("_all") for name in MODULE["INDEPENDENT_CASES"])


@pytest.mark.parametrize("action", ["run", "perf", "trace"])
def test_matrix_uses_matching_q8_diagnostics_only_for_trace(action, capsys):
    MODULE["main"]([action, "--dry-run", "--config", "/tmp/host settings.json"])
    output = capsys.readouterr().out
    assert output.count("[CASE]") == (4 if action == "trace" else 8)
    assert "q0_all" not in output and "q8_all" not in output
    if action == "trace":
        assert "q8_" not in output
    for name in MODULE["INDEPENDENT_CASES"]:
        args = MODULE["case_arguments"](action, name)
        assert ("--combine-ready-trace" in args) == (action == "trace" and name.startswith("q8_"))
        assert ("--timer-ranks" in args) == (action == "trace")


def fake_launcher(monkeypatch, tmp_path, launch):
    config = tmp_path / "host.json"
    config.write_text(json.dumps(dict(case_dir=str(tmp_path / "results"), node_rank=0)))
    monkeypatch.setattr(
        MODULE["main"].__globals__["runpy"],
        "run_path",
        lambda _: {
            "main": lambda args, **kwargs: launch(args),
            "DEFAULT_CONFIG": config,
            "validate": lambda _: None,
            "print_trace_parts": lambda *args, **kwargs: None,
        },
    )


def write_result(path, options, low=101.25, high=202.5, action="perf"):
    path.mkdir(parents=True)
    (path / "run.json").write_text(
        json.dumps(dict(action=action, warmup=20, iters=100, perf_method="legacy", **options))
    )
    (path / "launch.json").write_text(json.dumps(dict(node_rank=0)))
    if action == "perf":
        (path / "summary.csv").write_text(f"bs,mean_us,rank_max_mean_us\n32,{low},{high}\n")
    return path


def test_matrix_stops_at_first_failure(monkeypatch, tmp_path, capsys):
    launches = []

    def fail(args):
        launches.append(args)
        raise RuntimeError("accuracy failed")

    fake_launcher(monkeypatch, tmp_path, fail)
    with pytest.raises(RuntimeError, match="accuracy failed"):
        MODULE["main"](["run"])
    assert len(launches) == 1 and launches[0] == ["run", *MODULE["CASES"]["q0_scan"]]
    output = capsys.readouterr().out
    assert "BEGIN MEGAMOE PIPELINE SUMMARY" in output and "accuracy failed" in output
    summary = next((tmp_path / "results").glob("pipeline_run_*/summary.json"))
    rows = json.loads(summary.read_text())
    assert [row["status"] for row in rows] == ["FAILED", *["NOT_RUN"] * 7]
    assert not any("mean_us" in row for row in rows)


@pytest.mark.parametrize("action", ["run", "perf", "trace"])
def test_matrix_summarizes_exact_returned_directories(monkeypatch, tmp_path, capsys, action):
    launches = []
    # Unrelated historical results must never be mistaken for this invocation.
    write_result(
        tmp_path / "results" / "perf_node_0_newest", dict(dispatch_workers=14, combine_sparse_scan=True), low=1, high=2
    )

    def launch(args):
        index = len(launches)
        launches.append(args)
        options = {
            "combine_reduce_qps": int(args[args.index("--combine-reduce-qps") + 1]),
            "dispatch_workers": int(args[args.index("--dispatch-workers") + 1]),
            **{key: "--" + key.replace("_", "-") in args for key in MODULE["SWITCHES"]},
        }
        return write_result(tmp_path / "results" / f"run {index}", options, 101.25 + index, 202.5 + index, action)

    fake_launcher(monkeypatch, tmp_path, launch)
    directory = MODULE["main"]([action])
    rows = json.loads((directory / "summary.json").read_text())
    selected = [name for name in MODULE["INDEPENDENT_CASES"] if action != "trace" or name.startswith("q0_")]
    assert [row["case"] for row in rows] == selected
    assert all(row["status"] == "OK" for row in rows)
    assert [row["run_dir"] for row in rows] == [
        str((tmp_path / "results" / f"run {i}").resolve()) for i in range(len(selected))
    ]
    if action == "perf":
        assert [row["mean_us"] for row in rows] == [101.25 + i for i in range(8)]
        assert [row["rank_max_mean_us"] for row in rows] == [202.5 + i for i in range(8)]
    else:
        assert not any("mean_us" in row for row in rows)
    output = capsys.readouterr().out
    assert output.count("BEGIN MEGAMOE PIPELINE SUMMARY") == 1
    assert (directory / "summary.txt").read_text(encoding="utf-8").strip() in output
    assert (directory / "summary.csv").is_file()


def test_partial_results_survive_failure(monkeypatch, tmp_path):
    launches = []

    def launch(args):
        launches.append(args)
        if len(launches) == 2:
            raise RuntimeError("second case failed")
        return write_result(tmp_path / "results" / "first", dict(dispatch_workers=14, combine_sparse_scan=True))

    fake_launcher(monkeypatch, tmp_path, launch)
    with pytest.raises(RuntimeError, match="second case failed"):
        MODULE["main"](["perf", "--suite", "q0"])
    assert len(launches) == 2
    rows = json.loads(next((tmp_path / "results").glob("pipeline_perf_*/summary.json")).read_text())
    assert [row["status"] for row in rows] == ["OK", "FAILED", "NOT_RUN", "NOT_RUN"]
    assert rows[0]["rank_max_mean_us"] == 202.5


def test_existing_results_identify_cases_from_configuration(monkeypatch, tmp_path):
    sync = write_result(
        tmp_path / "a",
        dict(combine_reduce_qps=8, dispatch_workers=28, reduce_lane_init=True, combine_sync_progress=True),
    )
    scan = write_result(tmp_path / "b", dict(dispatch_workers=14, combine_sparse_scan=True), low=130, high=150)
    monkeypatch.setattr(
        MODULE["main"].__globals__["runpy"], "run_path", lambda _: pytest.fail("Summary must not launch")
    )
    directory = MODULE["main"](["summary", str(sync), str(scan)])
    rows = json.loads((directory / "summary.json").read_text())
    assert [row["case"] for row in rows] == ["custom", "q0_scan"]
    assert [row["rank_max_mean_us"] for row in rows] == [202.5, 150]
    assert rows[0]["combine_sync_progress"] is True
    assert "combine_sync_progress" in (directory / "summary.txt").read_text()


def test_wrong_case_or_incomplete_metrics_are_rejected(tmp_path):
    path = write_result(tmp_path / "run", dict(dispatch_workers=14, combine_sparse_scan=True))
    with pytest.raises(ValueError, match="does not match"):
        MODULE["read_result"](path, "q8_dispatch14")
    (path / "summary.csv").unlink()
    with pytest.raises(FileNotFoundError):
        MODULE["read_result"](path, "q0_scan")


@pytest.mark.parametrize(
    "retired", ["combine_async", "combine_owner_half", "combine_sync_progress", "dispatch_unknown"]
)
def test_retired_flags_cannot_be_relabelled_as_current_baseline(tmp_path, retired):
    path = write_result(tmp_path / "old", dict(dispatch_workers=14, combine_sparse_scan=True, **{retired: True}))
    assert MODULE["read_result"](path)["case"] == "custom"
    with pytest.raises(ValueError, match="does not match"):
        MODULE["read_result"](path, "q0_scan")


def test_trace_prints_one_collection_after_all_cases(monkeypatch, tmp_path):
    calls = []
    selected = [name for name in MODULE["INDEPENDENT_CASES"] if name.startswith("q0_")]
    config = tmp_path / "host.json"
    config.write_text(json.dumps(dict(case_dir=str(tmp_path / "results"), node_rank=0)))

    def launch(args, *, print_text=True):
        assert print_text is False
        name = selected[len(calls)]
        calls.append(name)
        return write_result(tmp_path / name, MODULE["expected_settings"](name, "trace"), action="trace")

    def show(directory, config, *, pipeline):
        assert pipeline and calls == selected
        assert all(row["status"] == "OK" for row in json.loads((directory / "summary.json").read_text()))
        calls.append("printed")

    monkeypatch.setattr(
        MODULE["main"].__globals__["runpy"],
        "run_path",
        lambda _: dict(main=launch, DEFAULT_CONFIG=config, validate=lambda _: None, print_trace_parts=show),
    )
    MODULE["main"](["trace"])
    assert calls[-1] == "printed"


@pytest.mark.parametrize("suite,count", [("q0", 4), ("q8", 4), ("all", 8)])
def test_trace_explicit_suite_overrides_q0_default(suite, count, capsys):
    MODULE["main"](["trace", "--suite", suite, "--dry-run"])
    output = capsys.readouterr().out
    assert output.count("[CASE]") == count
    assert ("q8_" in output) == (suite in ("all", "q8"))
