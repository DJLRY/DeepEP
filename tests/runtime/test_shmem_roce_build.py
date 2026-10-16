# Copyright (c) 2026, Lu Lu

"""SDK backend selection and reuse checks without CANN or a network build."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("backend", ["NONE", "XSCALE", "HNS_1825"])
def test_build_passes_and_records_rdma_backend(tmp_path, monkeypatch, backend):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = load(ROOT / "scripts/build_shmem.py")
    source = tmp_path / "third-party/shmem"
    (source / "scripts").mkdir(parents=True)
    (source / ".git").touch()
    (source / "scripts/build.sh").touch()
    lock = {"shmem": {"revision": "pinned"}, "cann_version": "test", "target": "Ascend950"}
    (tmp_path / "dependencies.lock.json").write_text(json.dumps(lock))
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module, "cann_version", lambda expected: {"Version": expected})
    monkeypatch.setattr(module, "source_patches", lambda path: [])
    monkeypatch.setattr(module.shutil, "which", lambda name: sys.executable)
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda command, **kwargs: str(source) if command[-1] == "--show-toplevel" else "pinned",
    )
    calls, identities = [], []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(
        module, "record", lambda sdk, revision, **kwargs: identities.append(kwargs) or sdk / "deepep-sdk.json"
    )
    module.prepare(tmp_path, rdma_backend=backend)
    assert len(calls) == 1
    assert calls[0][5:] == ([] if backend == "NONE" else ["-enable_rdma", "-rdma_backend", backend])
    assert identities[0]["build_info"]["rdma_backend"] == backend


@pytest.mark.parametrize("backend", ["NONE", "XSCALE", "HNS_1825"])
def test_rdma_backend_participates_in_build_cache(backend):
    module = load(ROOT / "build_support.py")
    sdk = {"build": {"rdma_backend": backend}}
    assert module.shmem_rdma_backend(sdk) == backend
    if backend != "NONE":
        assert module.build_cache_key({"shmem": sdk}) != module.build_cache_key(
            {"shmem": {"build": {"rdma_backend": "NONE"}}}
        )
    assert module.shmem_rdma_backend({}) == "NONE"


@pytest.mark.parametrize("value", ["IN_DIE", "auto", None, 1, []])
def test_invalid_sdk_rdma_backend_rejected(value):
    module = load(ROOT / "build_support.py")
    with pytest.raises(RuntimeError, match="rdma_backend"):
        module.shmem_rdma_backend({"build": {"rdma_backend": value}})


@pytest.mark.parametrize(
    "recorded,requested,matches",
    [(None, "NONE", True), ("XSCALE", "XSCALE", True), ("XSCALE", "HNS_1825", False), ("NONE", "XSCALE", False)],
)
def test_sdk_reuse_checks_rdma_backend(tmp_path, monkeypatch, recorded, requested, matches):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = load(ROOT / "scripts/build_shmem.py")
    source = tmp_path / "third-party/shmem"
    (source / "scripts").mkdir(parents=True)
    (source / ".git").touch()
    (source / "scripts/build.sh").touch()
    lock = {"shmem": {"revision": "pinned"}, "cann_version": "test", "target": "Ascend950"}
    (tmp_path / "dependencies.lock.json").write_text(json.dumps(lock))
    manifest = source / "install/shmem/deepep-sdk.json"
    manifest.parent.mkdir(parents=True)
    build = {"cann_version": "test", "cann_timestamp": "unknown", "target": "Ascend950", "source_patches": []}
    if recorded is not None:
        build["rdma_backend"] = recorded
    manifest.write_text(json.dumps({"revision": "pinned", "build": build}))
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module, "cann_version", lambda expected: {"Version": expected})
    monkeypatch.setattr(module, "source_patches", lambda path: [])
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda command, **kwargs: str(source) if command[-1] == "--show-toplevel" else "pinned",
    )
    validated = []
    monkeypatch.setitem(
        sys.modules, "build_support", SimpleNamespace(validate_shmem_sdk=lambda *args: validated.append(args))
    )
    if matches:
        assert module.prepare(tmp_path, if_needed=True, rdma_backend=requested) == manifest
        assert validated == [(manifest.parent, lock["shmem"])]
    else:
        with pytest.raises(RuntimeError, match="identity changed"):
            module.prepare(tmp_path, if_needed=True, rdma_backend=requested)
        assert not validated
