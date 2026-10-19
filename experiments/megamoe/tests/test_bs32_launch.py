# Copyright (c) 2026, Lu Lu

"""Check persisted settings and build/run separation without CANN or devices."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("bs32_launcher", ROOT / "experiments/megamoe/scripts/bs32.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def saved(tmp_path):
    cann = tmp_path / "CANN with spaces" / "set_env.sh"
    cann.parent.mkdir()
    cann.write_text("export ASCEND_HOME_PATH=/opt/cann\n")
    config = dict(
        launcher.defaults(),
        node_rank=0,
        master_addr="192.0.2.10",
        cann_env=str(cann),
        rdma_backend="HNS_1825",
        case_dir=str(tmp_path / "results"),
    )
    path = tmp_path / "node.json"
    launcher.save_config(path, config)
    return path, config


def test_first_configuration_and_explicit_partial_update(tmp_path, saved):
    original_path, original = saved
    path = tmp_path / "other-host.json"
    launcher.main(
        [
            "configure",
            "--config",
            str(path),
            "--node-rank",
            "1",
            "--master-addr",
            original["master_addr"],
            "--cann-env",
            original["cann_env"],
            "--rdma-backend",
            "XSCALE",
        ]
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["node_rank"] == 1
    assert json.loads(original_path.read_text(encoding="utf-8")) == original
    launcher.main(["configure", "--config", str(path), "--master-addr", "192.0.2.20"])
    assert json.loads(path.read_text(encoding="utf-8")) == dict(config, master_addr="192.0.2.20")


def test_repeated_run_reuses_settings_without_prompt_or_environment_override(saved, monkeypatch):
    path, config = saved
    before = path.read_bytes()
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("Prompt on repeat run"))
    monkeypatch.setenv("MASTER_ADDR", "192.0.2.99")
    monkeypatch.setenv("NODE_RANK", "1")
    calls = []
    monkeypatch.setattr(launcher, "dispatch", lambda *a, **kw: calls.append((a, kw)))
    for _ in range(2):
        launcher.main(["run", "--config", str(path)])
    assert calls == [(("run", config), {"rebuild_shmem": False})] * 2
    assert path.read_bytes() == before


def test_interactive_config_keeps_existing_values_on_enter(saved, monkeypatch):
    path, config = saved
    monkeypatch.setattr(launcher.sys.stdin, "isatty", lambda: True)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda label: prompts.append(label) or "")
    launcher.main(["configure", "--config", str(path)])
    assert len(prompts) == 4
    assert json.loads(path.read_text(encoding="utf-8")) == config


@pytest.mark.parametrize("failure", ["preflight", "golden"])
def test_failed_preparation_never_launches_benchmark(saved, monkeypatch, failure):
    _, config = saved
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher, "environment", lambda _: {})
    calls = []

    def preflight(*args):
        if failure == "preflight":
            raise RuntimeError("missing installation")

    def execute(command, *args, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(launcher, "preflight", preflight)
    monkeypatch.setattr(launcher, "execute", execute)
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        launcher.dispatch("run", config)
    assert len(calls) == (1 if failure == "golden" else 0)
    assert all("torch.distributed.run" not in command for command in calls)


def test_isolated_cli_help():
    result = subprocess.run([sys.executable, "-I", launcher.__file__, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "configure,show-config,build,golden,run" in result.stdout


@pytest.mark.parametrize(
    "options",
    [
        ["--node-rank", "2"],
        ["--master-addr", "127.0.0.1"],
        ["--master-port", "26733"],
        ["--golden-workers", "0"],
        ["--rdma-backend", "NONE"],
        ["--cann-env", "/missing/set_env.sh"],
    ],
)
def test_bad_edit_preserves_previous_configuration(saved, options):
    path, _ = saved
    before = path.read_bytes()
    with pytest.raises(ValueError):
        launcher.main(["configure", "--config", str(path), *options])
    assert path.read_bytes() == before


def test_runtime_cannot_mutate_config_or_rebuild_sdk(saved):
    path, _ = saved
    for options in (["--master-addr", "192.0.2.20"], ["--rebuild-shmem"]):
        with pytest.raises(SystemExit):
            launcher.main(["run", "--config", str(path), *options])


@pytest.mark.parametrize("action,expected", [("build", 1), ("golden", 1), ("run", 2)])
def test_stage_separation_and_fixed_case(saved, monkeypatch, action, expected):
    _, config = saved
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher, "environment", lambda _: {"TEST": "yes"})
    probes = []
    monkeypatch.setattr(launcher, "preflight", lambda *args: probes.append(args))
    calls = []
    monkeypatch.setattr(launcher, "execute", lambda *args, **kwargs: calls.append((args, kwargs)))
    launcher.dispatch(action, config, rebuild_shmem=action == "build")
    assert len(calls) == expected
    if action == "build":
        assert calls[0][0][0] == [
            "bash",
            str(ROOT / "experiments/megamoe/scripts/build.sh"),
            "--rdma-backend",
            "HNS_1825",
            "--rebuild-shmem",
        ]
        assert not probes
        assert not Path(config["case_dir"]).exists()
    else:
        assert len(probes) == 1
        golden = calls[0][0][0]
        assert golden[:4] == [config["python"], "-I", "-m", "deep_ep_experimental.megamoe.golden"]
        assert golden[golden.index("--ranks") + 1] == "0-15"
        case_path = Path(golden[golden.index("--config") + 1])
        assert json.loads(case_path.read_text(encoding="utf-8")) == launcher.CASE
        assert launcher.CASE["bs_values"] == [32]
        assert launcher.CASE["model"] == dict(
            hidden=3584, intermediate_hidden=1280, num_experts=384, num_topk=8, num_shared_experts=0
        )
        if action == "run":
            command = calls[1][0][0]
            assert command[:4] == [config["python"], "-I", "-m", "torch.distributed.run"]
            assert command[command.index("--nproc-per-node") + 1] == "16"
            assert command[command.index("--golden-dir") + 1] == golden[golden.index("--golden-dir") + 1]
            assert command[command.index("--config") + 1] == str(case_path)


def test_second_host_rank_range_and_saved_python(saved):
    _, config = saved
    config = dict(config, node_rank=1, python="/opt/env with spaces/bin/python")
    golden, benchmark = launcher.test_commands(config, Path(config["case_dir"]))
    assert golden[golden.index("--ranks") + 1] == "16-31"
    assert benchmark[benchmark.index("--node-rank") + 1] == "1"
    assert golden[0] == benchmark[0] == config["python"]


def test_environment_uses_positional_paths_and_removes_import_overrides(saved, monkeypatch):
    _, config = saved
    for name in ("SHMEM_ROOT", "CATLASS_ROOT", "PYTHONPATH", "PYTHONHOME", "PYTHON_EXEC"):
        monkeypatch.setenv(name, "unwanted")
    before = dict(os.environ)

    def source(command, *, env, cwd, check, text, stdout):
        assert command[0:2] == ["bash", "-c"]
        assert command[4:6] == [config["cann_env"], config["python"]]
        assert config["cann_env"] not in command[2]
        assert "PYTHONPATH" not in env and "SHMEM_ROOT" not in env
        return subprocess.CompletedProcess(command, 0, json.dumps(dict(env, ASCEND_HOME_PATH="/opt/cann")))

    monkeypatch.setattr(launcher.subprocess, "run", source)
    env = launcher.environment(config)
    assert env["MASTER_ADDR"] == config["master_addr"]
    assert env["PYTHONSAFEPATH"] == "1"
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == ",".join(map(str, range(16)))
    assert env["DEEPEP_SHMEM_ENDPOINT"] == "tcp://192.0.2.10:26733"
    assert env["DEEPEP_MEGAMOE_TIMER"] == "OFF"
    assert dict(os.environ) == before


def test_logged_child_failure_is_not_reported_as_success(tmp_path):
    log = tmp_path / "failure.log"
    with pytest.raises(subprocess.CalledProcessError) as error:
        launcher.execute(
            [sys.executable, "-I", "-c", "print('test failure'); raise SystemExit(7)"],
            dict(os.environ),
            log,
            cwd=tmp_path,
        )
    assert error.value.returncode == 7
    assert "test failure" in log.read_text(encoding="utf-8")
