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
        "q0_scan": ("q0_base", {"--combine-sparse-scan"}),
        "q0_async": ("q0_base", {"--combine-async"}),
        "q8_owner_half": ("q8_base", {"--combine-owner-half"}),
        "q8_sync": ("q8_base", {"--combine-sync-progress"}),
    }
    for name, (baseline, additions) in pairs.items():
        assert set(cases[name]) - set(cases[baseline]) == additions
        assert set(cases[baseline]) - set(cases[name]) == set()
    assert cases["q8_dispatch14"] == tuple("14" if arg == "28" else arg for arg in cases["q8_base"])
    assert len(MODULE["INDEPENDENT_CASES"]) == 7
    assert not any(name.endswith("_all") for name in MODULE["INDEPENDENT_CASES"])


@pytest.mark.parametrize("action", ["run", "perf", "trace"])
def test_matrix_uses_matching_q8_diagnostics_only_for_trace(action, capsys):
    MODULE["main"]([action, "--dry-run", "--config", "/tmp/host settings.json"])
    output = capsys.readouterr().out
    assert output.count("[CASE]") == 7 and "q0_all" not in output and "q8_all" not in output
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
        lambda _: {"main": launch, "DEFAULT_CONFIG": config, "validate": lambda _: None},
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
    assert len(launches) == 1 and launches[0] == ["run", *MODULE["CASES"]["q0_base"]]
    output = capsys.readouterr().out
    assert "BEGIN MEGAMOE PIPELINE SUMMARY" in output and "accuracy failed" in output
    summary = next((tmp_path / "results").glob("pipeline_run_*/summary.json"))
    rows = json.loads(summary.read_text())
    assert [row["status"] for row in rows] == ["FAILED", *["NOT_RUN"] * 6]
    assert not any("mean_us" in row for row in rows)


@pytest.mark.parametrize("action", ["run", "perf", "trace"])
def test_matrix_summarizes_exact_returned_directories(monkeypatch, tmp_path, capsys, action):
    launches = []
    # Unrelated historical results must never be mistaken for this invocation.
    write_result(tmp_path / "results" / "perf_node_0_newest", dict(dispatch_workers=14), low=1, high=2)

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
    assert [row["case"] for row in rows] == list(MODULE["INDEPENDENT_CASES"])
    assert all(row["status"] == "OK" for row in rows)
    assert [row["run_dir"] for row in rows] == [str((tmp_path / "results" / f"run {i}").resolve()) for i in range(7)]
    if action == "perf":
        assert [row["mean_us"] for row in rows] == [101.25 + i for i in range(7)]
        assert [row["rank_max_mean_us"] for row in rows] == [202.5 + i for i in range(7)]
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
        return write_result(tmp_path / "results" / "first", dict(dispatch_workers=14))

    fake_launcher(monkeypatch, tmp_path, launch)
    with pytest.raises(RuntimeError, match="second case failed"):
        MODULE["main"](["perf", "--suite", "q0"])
    assert len(launches) == 2
    rows = json.loads(next((tmp_path / "results").glob("pipeline_perf_*/summary.json")).read_text())
    assert [row["status"] for row in rows] == ["OK", "FAILED", "NOT_RUN"]
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
    assert [row["case"] for row in rows] == ["q8_sync", "q0_scan"]
    assert [row["rank_max_mean_us"] for row in rows] == [202.5, 150]


def test_wrong_case_or_incomplete_metrics_are_rejected(tmp_path):
    path = write_result(tmp_path / "run", dict(dispatch_workers=14))
    with pytest.raises(ValueError, match="does not match"):
        MODULE["read_result"](path, "q8_base")
    (path / "summary.csv").unlink()
    with pytest.raises(FileNotFoundError):
        MODULE["read_result"](path, "q0_base")
