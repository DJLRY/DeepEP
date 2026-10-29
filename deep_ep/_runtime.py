# Copyright (c) 2026, Lu Lu

"""Shared V1/V2 resource owner; no EP algorithms are enabled by this module."""

import hashlib
import ipaddress
import os
import threading
import warnings
from contextlib import nullcontext
from pathlib import Path
from uuid import UUID

from ._native import npu_api, require_native
from ._store import StoreGroup

_OWNER = None
_LOCK = threading.Lock()


def integer(name, value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def endpoint():
    value = os.environ.get("DEEPEP_SHMEM_ENDPOINT", "")
    if not value.startswith("tcp://") or len(value) >= 64:
        raise ValueError("Set DEEPEP_SHMEM_ENDPOINT=tcp://<IPv4>:<unused-port>")
    host, separator, port = value[6:].partition(":")
    if not separator or not port.isdecimal() or not 1024 <= int(port) <= 65535:
        raise ValueError("DEEPEP_SHMEM_ENDPOINT requires an unprivileged TCP port")
    ipaddress.IPv4Address(host)
    return value


def _host_boot_id():
    # hostname 可能在不同机器上重复；Linux boot_id 在本机各进程间一致。
    try:
        return UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip()).hex
    except (OSError, ValueError):
        # 先参加配置交换，让所有 rank 一起失败，不能退回不可靠的 hostname。
        return ""


def rank_and_world(group, rank=None, world_size=None):
    """Resolve an explicit SHMEM identity or an existing framework group."""
    if group is None:
        from .buffers._elastic_layout import WORLD_SIZES

        integer("world_size", world_size, 2, 512)
        if world_size not in WORLD_SIZES:
            raise ValueError("Runtime supports EP2/4/8/16/32/64/128/256/512")
        integer("rank", rank, 0, world_size - 1)
        return rank, world_size
    if rank is not None or world_size is not None:
        raise ValueError("Pass either group or rank/world_size, not both")
    import torch.distributed as dist

    if not isinstance(group, dist.ProcessGroup):
        raise TypeError("group must be an initialized torch.distributed ProcessGroup")
    return dist.get_rank(group), dist.get_world_size(group)


class Runtime:
    """One explicit collective lifetime per process; a live owner is retained."""

    _megamoe = False

    def __init__(
        self,
        group,
        capacity,
        timeout=120,
        *,
        udma=False,
        rank=None,
        world_size=None,
        bootstrap="default",
        combine_only=False,
        ranks_per_node=0,
    ):
        global _OWNER
        native = require_native("Runtime.__init__")
        integer("capacity", capacity, 2 << 20, (1 << 30) if self._megamoe else (1 << 63) - 1)
        if capacity % (2 << 20):
            raise ValueError("capacity must be aligned to 2 MiB")
        integer("timeout", timeout, 1, 3600)
        if bootstrap not in ("default", "mpi"):
            raise ValueError("bootstrap must be default or mpi")
        if bootstrap == "mpi":
            if group is not None or not udma or self._megamoe:
                raise ValueError("MPI bootstrap requires explicit rank/world_size and the Elastic UDMA runtime")
            if not callable(getattr(native, "create_mpi_runtime", None)):
                raise RuntimeError("Rebuild/install DeepEP with MPI bootstrap support")
            try:
                mpi_rank, mpi_size = (int(os.environ[name]) for name in ("PMI_RANK", "PMI_SIZE"))
            except (KeyError, ValueError) as exc:
                raise ValueError("MPI bootstrap requires an MPICH launcher with PMI_RANK/PMI_SIZE") from exc
            if (rank, world_size) != (mpi_rank, mpi_size):
                raise ValueError("Explicit rank/world_size disagree with MPI_COMM_WORLD")
        address = "" if bootstrap == "mpi" else endpoint()
        if type(combine_only) is not bool or (combine_only and not udma):
            raise ValueError("combine_only requires a UDMA runtime and a bool value")
        self.combine_only = combine_only
        dense_qps = os.environ.get("DEEPEP_SHMEM_DENSE_QPS", "0")
        if dense_qps not in ("0", "1"):
            raise ValueError("DEEPEP_SHMEM_DENSE_QPS must be 0 or 1")
        self.bootstrap = bootstrap
        store_group = self._megamoe and isinstance(group, StoreGroup)
        if self._megamoe and (group is None or udma):
            raise ValueError("MegaMoE requires a ProcessGroup or StoreGroup and its dedicated runtime")
        if store_group:
            if rank is not None or world_size is not None:
                raise ValueError("Pass either group or rank/world_size, not both")
            rank, size = group.rank(), group.size()
        else:
            rank, size = rank_and_world(group, rank, world_size)
        factory = native.Runtime
        if self._megamoe:
            integer("world_size", size, 2, 128)
            integer("ranks_per_node", ranks_per_node, 0, size)
            factory = getattr(native, "create_megamoe_runtime", None)
            if factory is None:
                raise RuntimeError("Rebuild with DEEPEP_BUILD_MEGAMOE=ON")
        else:
            from .buffers._elastic_layout import WORLD_SIZES

            if size not in WORLD_SIZES:
                raise ValueError("Runtime supports EP2/4/8/16/32/64/128/256/512")
        self.use_roce = self._megamoe and ranks_per_node == 16 and size > 16
        if group is not None and not store_group:
            import torch
            import torch.distributed as dist

            backend = str(dist.get_backend(group))
            if backend not in {"gloo", "hccl"}:
                raise NotImplementedError("Runtime bootstrap supports Gloo or HCCL process groups")
        npu = npu_api()
        self.device = npu.current_device()
        self.group = group
        self.low_latency_comm_args = None
        self.closed = False
        with _LOCK:
            busy = _OWNER is not None or native.runtime_busy()

            if group is None:
                if busy:
                    raise RuntimeError("This process already owns a SHMEM runtime")
            else:

                def digest(value):
                    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:7], "little")

                local_values = [capacity, timeout, digest(address), int(busy)]
                if self._megamoe:
                    host_id = _host_boot_id()
                    local_values.extend(
                        [
                            digest(host_id) if host_id else 0,
                            digest(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")),
                            self.device,
                            ranks_per_node,
                            {"XSCALE": 1, "HNS_1825": 2}.get(getattr(native, "shmem_rdma_backend", "NONE"), 0)
                            if getattr(native, "shmem_sdk_matches", False)
                            else 0,
                        ]
                    )
                local_values.append(int(udma) + 2 * int(combine_only) + 4 * int(dense_qps) + 8 * int(self.use_roce))
                if store_group:
                    values = group.all_gather(local_values, "runtime configuration")
                else:
                    local = torch.tensor(
                        local_values, dtype=torch.int64, device=f"npu:{self.device}" if backend == "hccl" else "cpu"
                    )
                    peers = [torch.empty_like(local) for _ in range(size)]
                    dist.all_gather(peers, local, group=group)
                    values = [value.cpu().tolist() for value in peers]
                if any(value[3] for value in values):
                    raise RuntimeError("A rank already owns a SHMEM runtime; destroy it collectively first")
                if any(value[:3] != values[0][:3] for value in values):
                    raise ValueError("Ranks disagree on workspace capacity, timeout or bootstrap endpoint")
                if any(value[-1] != values[0][-1] for value in values):
                    raise ValueError("Ranks disagree on MTE/URMA/RoCE runtime engines or QP configuration")
                if self._megamoe:
                    if any(value[7] != values[0][7] for value in values):
                        raise ValueError("Ranks disagree on ranks_per_node")
                    invalid_hosts = [i for i, value in enumerate(values) if value[4] == 0]
                    if invalid_hosts:
                        raise RuntimeError(
                            f"Cannot read a valid Linux /proc/sys/kernel/random/boot_id on group ranks {invalid_hosts}; "
                            "host identity is required before SHMEM initialization"
                        )
                    hosts = {value[4] for value in values}
                    for host in hosts:
                        local = [value for value in values if value[4] == host]
                        if len({value[5] for value in local}) != 1:
                            raise ValueError("Ranks on one host must use a common visible-device mapping")
                    devices = {}
                    for peer_rank, value in enumerate(values):
                        key = (value[4], value[6])
                        if key in devices:
                            raise ValueError(
                                f"Each rank must use a distinct local NPU: group ranks {devices[key]} and "
                                f"{peer_rank} use local device {value[6]} on the same Linux host"
                            )
                        devices[key] = peer_rank
                    if ranks_per_node == 16:
                        if size not in (16, 32):
                            raise ValueError("MegaMoE with ranks_per_node=16 supports EP16 or two-host EP32")
                        host_ranks = [[i for i, value in enumerate(values) if value[4] == host] for host in hosts]
                        expected = [list(range(start, start + 16)) for start in range(0, size, 16)]
                        if sorted(host_ranks) != expected:
                            raise ValueError("ranks_per_node=16 requires contiguous groups of 16 ranks per Linux host")
                    if self.use_roce:
                        if any(value[8] == 0 for value in values):
                            raise RuntimeError(
                                "MegaMoE cross-host ranks_per_node=16 requires an XSCALE or HNS_1825 SHMEM SDK "
                                "matching the wheel; rebuild SHMEM with --rdma-backend and rebuild the wheel"
                            )
                        if any(value[8] != values[0][8] for value in values):
                            raise ValueError("Ranks disagree on SHMEM RDMA backend")
            self.rank, self.world_size = rank, size
            self.capacity = capacity
            self.notify_lock = threading.RLock()
            self.notify_failed = False
            self.notify_inflight = None
            self.stream = npu.Stream(device=self.device)
            self.native = (
                native.create_combine_runtime(rank, size, capacity, address, timeout, bootstrap == "mpi")
                if combine_only
                else native.create_mpi_runtime(rank, size, capacity, timeout)
                if bootstrap == "mpi"
                else (
                    native.Runtime(rank, size, capacity, address, timeout, udma)
                    if udma
                    else factory(rank, size, capacity, address, timeout, True)
                    if self.use_roce
                    else factory(rank, size, capacity, address, timeout)
                )
            )
            _OWNER = self
            if capacity == 8 << 30 and 16 <= size <= 256:
                # LL peers must finish initialization before any payload writes.
                self.native.barrier()

    def ensure_alive(self):
        if self.closed:
            raise RuntimeError("Buffer runtime has been destroyed")
        if getattr(self, "roce_failed", False):
            raise RuntimeError("RoCE runtime synchronization or cleanup failed; restart the distributed job")
        if npu_api().current_device() != self.device:
            raise RuntimeError("Use Buffer on its creating NPU device")

    def _roce_collective(self, operation):
        # MegaMoE senders quiet QP1 before returning. Complete all compute and
        # communication streams before any rank can release symmetric memory.
        # The generic SHMEM WORLD barrier directly reads remote GM and cannot
        # synchronize peers reachable only over RoCE.
        stage = f"RoCE runtime {operation}"
        try:
            error = None
            try:
                self.ensure_alive()
                npu_api().synchronize(self.device)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            packet = [stage, error]
            if isinstance(self.group, StoreGroup):
                peers = self.group.all_gather(packet, stage)
            else:
                import torch.distributed as dist

                peers = [None] * self.world_size
                dist.all_gather_object(peers, packet, group=self.group)
            if any(peer[0] != stage for peer in peers):
                raise RuntimeError(f"RoCE runtime collective order mismatch: {peers}")
            errors = [(rank, peer[1]) for rank, peer in enumerate(peers) if peer[1] is not None]
            if errors:
                raise RuntimeError(f"{stage}: device synchronization failed on ranks {errors}")
            getattr(self.native, operation)(host_coordinated=True)
        except Exception as exc:
            self.roce_failed = True
            if isinstance(self.group, StoreGroup):
                self.group.abort(exc)
            raise

    def close(self):
        global _OWNER
        with getattr(self, "notify_lock", nullcontext()), _LOCK:
            if self.closed:
                return
            if getattr(self, "use_roce", False):
                self._roce_collective("close")
            else:
                self.ensure_alive()
                self.stream.synchronize()
                self.native.close()
            self.notify_inflight = None
            self.low_latency_comm_args = None
            self.closed = True
            _OWNER = None

    def barrier(self):
        if self.use_roce:
            with self.notify_lock:
                self._roce_collective("barrier")
            return
        self.ensure_alive()
        npu_api().current_stream(self.device).synchronize()
        self.stream.synchronize()
        self.native.barrier()


def runtime_of(buffer):
    runtime = getattr(buffer, "_runtime", None)
    if runtime is None:
        raise RuntimeError("Buffer runtime was not initialized")
    return runtime


def warn_unclosed(buffer):
    runtime = getattr(buffer, "_runtime", None)
    if runtime is not None and not runtime.closed:
        warnings.warn(
            "Buffer.destroy() must be called collectively before process-group teardown; "
            "resources are retained to avoid a collective in garbage collection",
            ResourceWarning,
            stacklevel=2,
        )
