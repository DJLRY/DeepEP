# Copyright (c) 2026, Lu Lu
"""Keep independent experiment pairs and timer diagnostics consistent."""

from pathlib import Path
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


def test_matrix_stops_at_first_failure(monkeypatch):
    launches = []

    def fail(args):
        launches.append(args)
        raise RuntimeError("accuracy failed")

    monkeypatch.setattr(MODULE["main"].__globals__["runpy"], "run_path", lambda _: {"main": fail})
    with pytest.raises(RuntimeError, match="accuracy failed"):
        MODULE["main"](["run"])
    assert len(launches) == 1 and launches[0] == ["run", *MODULE["CASES"]["q0_base"]]
