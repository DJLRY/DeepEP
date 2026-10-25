# Copyright (c) 2026, Lu Lu

"""Firmware mode identity, consumer definitions, and historical dependency overlays."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    path = ROOT / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", [None, "1", True, 2, -1])
def test_ambiguous_hns_sdk_is_rejected(mode):
    build = load("build_support.py")
    record = load("scripts/record_shmem_sdk.py")
    identity = {"rdma_backend": "HNS_1825", "hns_1825_pi_on_chip": mode}
    with pytest.raises(RuntimeError, match="hns_1825_pi_on_chip"):
        build.shmem_rdma_backend({"build": identity})
    with pytest.raises(ValueError, match="hns_1825_pi_on_chip"):
        record.validate_doorbell(identity)


def test_doorbell_modes_invalidate_object_cache():
    build = load("build_support.py")
    keys = []
    for mode in (0, 1):
        identity = {"rdma_backend": "HNS_1825", "hns_1825_pi_on_chip": mode}
        assert build.shmem_rdma_backend({"build": identity}) == "HNS_1825"
        keys.append(build.build_cache_key({"shmem": identity}))
    assert keys[0] != keys[1]


@pytest.mark.parametrize("mode", [0, 1, None])
def test_cmake_consumes_sdk_doorbell_and_rejects_missing_identity(tmp_path, mode):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for configure validation")
    source = tmp_path / "source"
    source.mkdir()
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    identity = {"rdma_backend": "HNS_1825"}
    if mode is not None:
        identity["hns_1825_pi_on_chip"] = mode
    (sdk / "deepep-sdk.json").write_text(json.dumps({"build": identity}))
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\nproject(test NONE)\n"
        f'set(SHMEM_ROOT "{sdk.as_posix()}")\n'
        f'include("{(ROOT / "cmake/ShmemDoorbell.cmake").as_posix()}")\n'
        "get_directory_property(definitions COMPILE_DEFINITIONS)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/flags.txt" "${definitions}\n${DEEPEP_SHMEM_DOORBELL_FLAGS}\n")\n'
    )
    binary = tmp_path / "build"
    binary.mkdir()
    # Same minimal cache used by build_shmem.py before upstream build.sh.
    if mode is not None:
        (binary / "CMakeCache.txt").write_text(f"ACLSHMEMI_HNS_1825_PI_ON_CHIP:STRING={mode}\n")
    result = subprocess.run([cmake, "-S", str(source), "-B", str(binary)], capture_output=True, text=True)
    if mode is None:
        assert result.returncode != 0
        assert "hns_1825_pi_on_chip" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    assert (binary / "flags.txt").read_text().splitlines() == [
        f"ACLSHMEMI_HNS_1825_PI_ON_CHIP={mode}",
        f"-DACLSHMEMI_HNS_1825_PI_ON_CHIP={mode}",
    ]
    bad = subprocess.run(
        [cmake, "-S", str(source), "-B", str(binary), f"-DACLSHMEMI_HNS_1825_PI_ON_CHIP={1 - mode}"],
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0 and "disagrees" in bad.stderr


def test_all_historical_commits_accept_dependency_overlay():
    history = load("scripts/prepare_shmem_history.py")
    if not (ROOT / ".git").exists():
        pytest.skip("Full Git history is required")
    refs = history.git("rev-list", "HEAD").splitlines()
    for ref in refs:
        revision, files = history.plan(ref)
        assert revision == ref
        expected = json.loads((ROOT / "dependencies.lock.json").read_text())["shmem"]
        assert json.loads(files["dependencies.lock.json"])["shmem"] == expected
        for name in ("build_support.py", "scripts/build_shmem.py", "scripts/record_shmem_sdk.py"):
            compile(files[name], name, "exec")
        assert not any(name.startswith("experiments/megamoe/csrc/") for name in files)
        assert not any(name.startswith("experiments/megamoe/scripts/") for name in files)


def test_history_worktree_preserves_commit_and_algorithms(tmp_path):
    history = load("scripts/prepare_shmem_history.py")
    source, output = tmp_path / "repo", tmp_path / "old-version"
    source.mkdir()

    def git(*args):
        return history.git(*args, root=source)

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    original = {}
    for name in (*history.TEXT_PATHS, *history.COPY_PATHS):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (ROOT / name).read_text(encoding="utf-8")
        if name == "csrc/runtime/session.cpp":
            content = content.replace(
                "attr.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_ENABLED;",
                'check(aclshmemx_set_shared_jetty(ACLSHMEM_DATA_OP_UDMA, true), "aclshmemx_set_shared_jetty");',
            )
        path.write_text(content, encoding="utf-8")
        original[name] = content
    algorithm = source / "experiments/megamoe/csrc/algorithm.cpp"
    algorithm.parent.mkdir(parents=True, exist_ok=True)
    algorithm.write_text("// old algorithm remains byte-for-byte identical\n")
    git("add", ".")
    git("update-index", "--add", "--cacheinfo", "160000," + "1" * 40 + ",third-party/shmem")
    git("commit", "-qm", "Historical version")
    revision = git("rev-parse", "HEAD")
    report = history.prepare(revision, output, root=source)
    assert history.git("rev-parse", "HEAD", root=output) == revision
    assert git("rev-parse", "HEAD") == revision
    assert (output / algorithm.relative_to(source)).read_bytes() == algorithm.read_bytes()
    assert report["original_commit"] == revision
    pin = json.loads((ROOT / "dependencies.lock.json").read_text())["shmem"]["revision"]
    assert history.git("ls-files", "--stage", "third-party/shmem", root=output).startswith(f"160000 {pin}")
    assert "attr.runtime_config.udma_shared_jetty" in (output / "csrc/runtime/session.cpp").read_text()
    with pytest.raises(ValueError, match="new directory"):
        history.prepare(revision, output, root=source)
    # Release only our temporary worktree registration; pytest owns its files.
    git("worktree", "remove", "--force", str(output))
