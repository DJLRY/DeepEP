# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Execute public wrappers using CPU storage doubles; this does not simulate the NPU kernel."""

import contextlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from deep_ep_experimental.megamoe import api
from deep_ep_experimental.megamoe.validation import collective_check


class Device(str):
    @property
    def type(self):
        return self.split(":")[0]


DTYPES = dict(
    uint8=np.uint8,
    int8=np.int8,
    int32=np.int32,
    int64=np.int64,
    float32=np.float32,
    bfloat16=np.float32,
    float8_e4m3fn=np.uint8,
    float8_e8m0fnu=np.uint8,
)


class Tensor:
    def __init__(self, data, dtype, device="npu:0"):
        self.data, self.dtype, self.device = np.asarray(data, dtype=DTYPES[dtype]), dtype, Device(device)
        self.requires_grad = False
        self.streams = []
        self.npu_format = 2
        self._version = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __getitem__(self, index):
        result = Tensor(self.data[index], self.dtype, self.device)
        result.npu_format = self.npu_format
        return result

    def view(self, dtype):
        return Tensor(self.data.view(DTYPES[dtype]), dtype, self.device)

    def reshape(self, shape):
        return Tensor(self.data.reshape(shape), self.dtype, self.device)

    def transpose(self, first, second):
        return Tensor(self.data.swapaxes(first, second), self.dtype, self.device)

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.data), self.dtype, self.device)

    def clone(self):
        return Tensor(self.data.copy(), self.dtype, self.device)

    def is_contiguous(self):
        return self.data.flags.c_contiguous

    def untyped_storage(self):
        base = self.data
        while isinstance(base.base, np.ndarray):
            base = base.base
        return SimpleNamespace(data_ptr=lambda: base.ctypes.data)

    def data_ptr(self):
        return self.data.ctypes.data

    def storage_offset(self):
        return (self.data_ptr() - self.untyped_storage().data_ptr()) // self.data.itemsize

    def stride(self):
        return tuple(n // self.data.itemsize for n in self.data.strides)

    def unsqueeze(self, axis):
        result = Tensor(np.expand_dims(self.data, axis), self.dtype, self.device)
        result.npu_format = self.npu_format
        return result

    def is_inference(self):
        return False

    def copy_(self, other, non_blocking=False):
        np.copyto(self.data, other.data)
        self._version += 1
        return self

    def record_stream(self, stream):
        self.streams.append(stream)

    def cpu(self):
        return self

    def tolist(self):
        return self.data.tolist()

    def add_(self, other):
        self.data += other.data
        self._version += 1
        return self


@pytest.fixture
def backend(monkeypatch):
    events = []

    class Group:
        def size(self):
            return 2

    def gather(output, value, **kwargs):
        output[:] = [value] * 2

    dist = SimpleNamespace(ProcessGroup=Group, get_world_size=lambda _: 2, all_gather_object=gather)
    torch = SimpleNamespace(
        **{name: name for name in DTYPES},
        device=Device,
        empty=lambda shape, dtype, device: Tensor(np.empty(shape, dtype=DTYPES[dtype]), dtype, device),
        zeros=lambda shape, dtype, device: Tensor(np.zeros(shape), dtype, device),
        empty_like=lambda t, dtype=None, device=None: Tensor(
            np.empty(t.shape, dtype=DTYPES[dtype or t.dtype]), dtype or t.dtype, device or t.device
        ),
        tensor=lambda data, dtype, device, pin_memory=False: Tensor(data, dtype, device),
        distributed=dist,
        inference_mode=lambda _: contextlib.nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)

    def format_cast(t, acl_format, *, customize_dtype, input_dtype):
        # 检查实际传入的协议，不能把所有 format_cast 参数都静默忽略。
        assert t.dtype == "uint8" and acl_format in (2, 29)
        if acl_format == 2:
            assert t.npu_format == 50
        assert customize_dtype == "float8_e4m3fn" and input_dtype == "float4_e2m1fn_x2"
        events.append(("format_cast", t))
        t = Tensor(t.data.copy(), t.dtype, t.device)
        t.npu_format = 2 if acl_format == 2 else 50  # FP4 打包成 uint8 后，输出码与请求的29不同。
        return t

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            get_npu_format=lambda t: t.npu_format, npu_format_cast=format_cast, float4_e2m1fn_x2="float4_e2m1fn_x2"
        ),
    )
    stream = SimpleNamespace(wait_stream=lambda _: events.append("wait"), synchronize=lambda: events.append("sync"))
    npu = SimpleNamespace(
        current_device=lambda: 0, current_stream=lambda _: stream, stream=lambda _: contextlib.nullcontext()
    )
    monkeypatch.setattr(api, "npu_api", lambda: npu)
    monkeypatch.setattr(
        api,
        "require_native",
        lambda _: SimpleNamespace(megamoe_workspace_address=lambda _: 4096, megamoe_combine_reduce_version=3),
    )

    class Runtime:
        def __init__(self, group, capacity, timeout, *, ranks_per_node=0):
            self.device, self.stream, self.native = 0, stream, SimpleNamespace(capacity=capacity)
            self.ranks_per_node = ranks_per_node
            self.closed = False

        def ensure_alive(self):
            if self.closed:
                raise RuntimeError("Buffer runtime has been destroyed")

        def close(self):
            events.append("close")
            self.closed = True

    monkeypatch.setattr(api, "_MegaRuntime", Runtime)

    class Plan:
        def __init__(self, example, *args):
            self.bs, self.workspace_bytes = example.shape[0], 4096
            self.options = args
            events.append(("prepare", self.bs))

        def forward(self, *args):
            events.append(("launch", args))
            output = args[-1]
            output.data.fill(3)
            return output, Tensor([2, 3], "int32")

    ext = SimpleNamespace(
        buffer_size=lambda *args: 2 << 20,
        Plan=Plan,
        ffts_config=lambda: 123,
        timer_enabled=True,
        timer_numel=64,
        small_batch_max_tokens=256,
        hierarchical_max_tokens=256,
    )
    monkeypatch.setattr(api, "_extension", lambda: ext)
    buffer = api.SymmBuffer(Group(), 4, 17, 2, 1024, 256)
    buffer.topk_idx.data[:] = [0, 1]
    buffer.topk_weights.data[:] = [0.4, 0.6]
    yield SimpleNamespace(buffer=buffer, events=events, ext=ext, dist=dist)
    buffer.destroy()


def weights():
    prepared = (
        (Tensor(np.zeros((2, 512, 512)), "uint8"), Tensor(np.full((2, 512, 16, 2), 127), "float8_e8m0fnu")),
        (Tensor(np.zeros((2, 1024, 128)), "uint8"), Tensor(np.full((2, 1024, 4, 2), 127), "float8_e8m0fnu")),
    )
    for weight, _ in prepared:
        weight.npu_format = 50
    return prepared


def fp8_weights():
    return (
        (Tensor(np.zeros((2, 512, 1024)), "float8_e4m3fn"), Tensor(np.full((2, 512, 16, 2), 127), "float8_e8m0fnu")),
        (Tensor(np.zeros((2, 1024, 256)), "float8_e4m3fn"), Tensor(np.full((2, 1024, 4, 2), 127), "float8_e8m0fnu")),
    )


@pytest.mark.parametrize("roce", [False, True])
def test_benchmark_helpers_use_live_owner_and_select_shmem_transport(backend, roce):
    log = []
    backend.ext.benchmark_sleep = lambda cycles: log.append(("sleep", cycles))
    backend.ext.benchmark_barrier = lambda selected: log.append(("barrier", selected))
    backend.buffer._hierarchical_roce = roce
    backend.buffer.benchmark_sleep(20_000_000)
    backend.buffer.benchmark_barrier()
    assert log == [("sleep", 20_000_000), ("barrier", roce)]
    assert not backend.events  # No host completion/teardown or MegaMoE call.
    backend.buffer._runtime.closed = True
    with pytest.raises(RuntimeError, match="destroyed"):
        backend.buffer.benchmark_barrier()
    assert len(log) == 2


def test_shmem_benchmark_detects_old_wheel_before_capture(backend):
    with pytest.raises(RuntimeError, match="rebuilding"):
        backend.buffer.benchmark_sleep(0)


@pytest.mark.parametrize("bf16", [False, True])
def test_public_workflow_output_stats_timer_and_preparation_reuse(backend, bf16):
    b = backend.buffer
    assert b.x.shape == (17, 1024) and b.x_sf.shape == (17, 32)
    original = b.x
    assert b.allocate_inputs() is b and b.x is original
    x = Tensor(np.ones((17, 1024)), "bfloat16") if bf16 else None
    b.validate_inputs(17, x=x)
    b.prepare(17)
    b.prepare(17)
    y, stats = Tensor(np.zeros((17, 1024)), "bfloat16"), Tensor([7, 11], "int32")
    assert api.fp8_fp4_mega_moe(y, *weights(), b, cumulative_local_expert_recv_stats=stats, x=x) is None
    np.testing.assert_array_equal(y.data, 3)
    np.testing.assert_array_equal(stats.data, [9, 14])
    timer = b.allocate_timer()
    assert not timer.data.any()
    api.fp8_fp4_mega_moe(y, *weights(), b, cumulative_local_expert_recv_stats=stats, x=x, timer=timer)
    np.testing.assert_array_equal(stats.data, [11, 17])
    launches = [event[1] for event in backend.events if isinstance(event, tuple) and event[0] == "launch"]
    assert launches[0][8] is None and launches[1][8] is timer
    assert (launches[0][1] is None) == bf16
    assert launches[0][-1] is y and launches[1][-1] is y
    assert (
        sum(event == ("prepare", 17) for event in backend.events if isinstance(event, tuple) and event[0] == "prepare")
        == 1
    )
    assert stats.streams == [b._runtime.stream, b._runtime.stream]
    assert b.workspace_bytes == 4096 and b.symmetric_buffer_bytes == 2 << 20
    b.prepare(1)
    assert b._plan.bs == 1 and b._plan_bs == 1


def test_output_and_stats_aliases_rejected_before_launch(backend):
    b = backend.buffer
    y = Tensor(np.zeros((17, 1024)), "bfloat16")
    with pytest.raises(ValueError, match="alias"):
        api.fp8_fp4_mega_moe(y, *weights(), b, cumulative_local_expert_recv_stats=b.topk_idx[0])
    with pytest.raises(ValueError, match="overlap"):
        api.fp8_fp4_mega_moe(y, *weights(), b, x=y)
    assert not any(isinstance(event, tuple) and event[0] == "launch" for event in backend.events)


@pytest.mark.parametrize("bad", [-1, 4, 0])
def test_collective_routing_preflight_rejects_bad_ids(backend, bad):
    backend.buffer.topk_idx.data[0, 1] = bad
    with pytest.raises(ValueError, match="rank=0.*token=0"):
        backend.buffer.validate_inputs(17)


def test_collective_remote_error_and_bs_mismatch(backend):
    def mismatched(output, value, **kwargs):
        output[:] = [value, (None, 96)]

    backend.dist.all_gather_object = mismatched
    with pytest.raises(ValueError, match="ranks disagree.*rank=1: 96"):
        collective_check(backend.buffer.group, "prepare", None, 17)

    def failed(output, value, **kwargs):
        output[:] = [value, ("out of memory", 17)]

    backend.dist.all_gather_object = failed
    with pytest.raises(ValueError, match="rank=1: out of memory"):
        collective_check(backend.buffer.group, "prepare", None, 17)


def test_timer_disabled_and_destroyed_buffer_fail_explicitly(backend):
    backend.ext.timer_enabled = False
    with pytest.raises(RuntimeError, match="DEEPEP_MEGAMOE_TIMER"):
        backend.buffer.allocate_timer()
    backend.buffer.destroy()
    with pytest.raises(RuntimeError, match="destroyed"):
        api.fp8_fp4_mega_moe(Tensor(np.zeros((17, 1024)), "bfloat16"), *weights(), backend.buffer)


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_fp4_and_packed_e8m0_are_reinterpreted_without_numeric_cast(backend, dtype):
    raw = np.arange(512 * 64, dtype=np.uint8).reshape(1, 512, 64)
    # Distinct row-dependent bytes expose both weight and scale column mismatches.
    row_ids = np.arange(512, dtype=np.uint16)
    raw[0, :, 0] = row_ids % 256
    raw[0, :, 1] = row_ids // 256
    sf = np.empty((1, 512, 4), dtype=np.uint8)
    sf[0, :, 0] = row_ids % 256
    sf[0, :, 1] = row_ids // 256
    sf[0, :, 2:] = np.array([129, 140], dtype=np.uint8)
    pair = Tensor(raw.view(DTYPES[dtype]), dtype), Tensor(sf.view(np.int32), "int32")
    order = np.array([*range(128), *range(256, 384), *range(128, 256), *range(384, 512)])
    prepared = api.transform_weights_for_mega_moe(pair, pair)
    for idx, (weight, scale) in enumerate(prepared):
        rows = order if idx == 0 else np.arange(512)
        np.testing.assert_array_equal(weight.data, raw[:, rows])
        np.testing.assert_array_equal(scale.data, sf[:, rows].reshape(1, 512, 2, 2))
        assert weight.dtype == "uint8" and scale.dtype == "float8_e8m0fnu"
        assert weight.npu_format == 50
    assert sum(isinstance(event, tuple) and event[0] == "format_cast" for event in backend.events) == 3
    planar_weight, planar_scale = prepared[0].select(False)
    np.testing.assert_array_equal(planar_weight.data, raw)
    np.testing.assert_array_equal(planar_scale.data, sf.reshape(1, 512, 2, 2))


def test_shared_fp4_preparation_retains_planar_weights_and_scales(backend):
    raw = np.arange(512 * 64, dtype=np.uint8).reshape(1, 512, 64)
    sf = np.arange(512 * 4, dtype=np.uint8).reshape(1, 512, 2, 2)
    pair = Tensor(raw, "uint8"), Tensor(sf, "float8_e8m0fnu")
    for weight, scale in api.transform_weights_for_mega_moe(pair, pair, shared=True):
        np.testing.assert_array_equal(weight.data, raw)
        np.testing.assert_array_equal(scale.data, sf)


@pytest.mark.parametrize("bf16", [False, True])
def test_routed_layout_switches_with_actual_bs_without_hot_path_cast(backend, bf16):
    b = backend.buffer
    b.num_max_tokens_per_rank = 4096
    b.x = Tensor(np.zeros((4096, 1024)), "float8_e4m3fn")
    b.x_sf = Tensor(np.zeros((4096, 32)), "float8_e8m0fnu")
    b.topk_idx = Tensor(np.tile([0, 1], (4096, 1)), "int32")
    b.topk_weights = Tensor(np.ones((4096, 2)), "float32")
    raw = weights()
    for weight, _ in raw:
        weight.npu_format = 2
    raw[0][0].data[:, :, 0] = np.arange(512, dtype=np.uint16) % 251
    w1, w2 = api.transform_weights_for_mega_moe(*raw)
    # Preparing weights must not share writable storage with the caller's raw scales.
    assert w1.select(False)[1].untyped_storage().data_ptr() != raw[0][1].untyped_storage().data_ptr()
    casts = sum(isinstance(e, tuple) and e[0] == "format_cast" for e in backend.events)
    for bs in (96, 256, 257, 4096, 96):
        output = Tensor(np.zeros((bs, 1024)), "bfloat16")
        x = Tensor(np.ones((bs, 1024)), "bfloat16") if bf16 else None
        api.fp8_fp4_mega_moe(output, w1, w2, b, x=x)
        launch = [e[1] for e in backend.events if isinstance(e, tuple) and e[0] == "launch"][-1]
        selected = w1.select(bs <= 256)
        assert launch[4] is selected[0] and launch[6] is selected[1]
        assert b._plan.bs == bs  # actual BS, not max capacity or per-expert M
    assert sum(isinstance(e, tuple) and e[0] == "format_cast" for e in backend.events) == casts
    with pytest.raises(ValueError, match="keep its returned l1 pair intact"):
        api.fp8_fp4_mega_moe(Tensor(np.zeros((257, 1024)), "bfloat16"), tuple(w1), w2, b)


@pytest.mark.parametrize("layout", [True, False])
@pytest.mark.parametrize("index", [0, 1])
def test_prepared_layout_mutation_is_not_silently_cached(backend, layout, index):
    raw = weights()
    for weight, _ in raw:
        weight.npu_format = 2
    w1, _ = api.transform_weights_for_mega_moe(*raw)
    tensor = w1.select(layout)[index]
    tensor.add_(Tensor(np.ones(tensor.shape), tensor.dtype))
    # Even requesting the other layout must fail rather than using an out-of-date copy.
    with pytest.raises(ValueError, match="modified in-place"):
        w1.select(not layout)


def test_fp4_transform_rejects_double_preparation(backend):
    pair = Tensor(np.zeros((1, 256, 64)), "uint8"), Tensor(np.ones((1, 256, 2, 2)), "float8_e8m0fnu")
    pair[0].npu_format = 50
    with pytest.raises(ValueError, match="already converted NZ"):
        api.transform_weights_for_mega_moe(pair, pair)


def test_routed_transform_rejects_incomplete_gate_up_pair(backend):
    pair = Tensor(np.zeros((1, 128, 64)), "uint8"), Tensor(np.ones((1, 128, 2, 2)), "float8_e8m0fnu")
    with pytest.raises(ValueError, match="divisible by 128"):
        api.transform_weights_for_mega_moe(pair, pair)


def test_fp8_weights_remain_nd_without_fp4_format_cast(backend):
    pair = (Tensor(np.zeros((1, 128, 128)), "float8_e4m3fn"), Tensor(np.full((1, 128, 2, 2), 127), "float8_e8m0fnu"))
    for weight, scale in api.transform_weights_for_mega_moe(pair, pair):
        assert weight is pair[0]
        np.testing.assert_array_equal(scale.data, pair[1].data)
    assert not any(isinstance(event, tuple) and event[0] == "format_cast" for event in backend.events)


@pytest.mark.parametrize("weight_format", [29, 50])
def test_prepared_fp4_nz_variants_reach_launch(backend, weight_format):
    w1, w2 = weights()
    w1[0].npu_format = w2[0].npu_format = weight_format
    api.fp8_fp4_mega_moe(Tensor(np.zeros((17, 1024)), "bfloat16"), w1, w2, backend.buffer)
    assert any(isinstance(event, tuple) and event[0] == "launch" for event in backend.events)


@pytest.mark.parametrize("name", ["l1_weights", "l2_weights", "shared_l1_weights", "shared_l2_weights"])
@pytest.mark.parametrize("weight_format", [2, 4, 51, 52, 53, 54])
def test_invalid_fp4_layout_rejected_with_tensor_context_before_launch(backend, name, weight_format):
    b = backend.buffer
    b.num_shared_experts = 2
    pairs = dict(zip(("l1_weights", "l2_weights", "shared_l1_weights", "shared_l2_weights"), (*weights(), *weights())))
    pairs[name][0].npu_format = weight_format
    with pytest.raises(ValueError, match=rf"{name}:.*got format={weight_format}, dtype=uint8, shape="):
        api.fp8_fp4_mega_moe(Tensor(np.zeros((17, 1024)), "bfloat16"), sym_buffer=b, **pairs)
    assert not any(isinstance(event, tuple) and event[0] == "launch" for event in backend.events)


def test_format_cast_returning_nd_fails_at_preparation(backend, monkeypatch):
    monkeypatch.setattr(sys.modules["torch_npu"], "npu_format_cast", lambda t, *a, **k: t)
    pair = (Tensor(np.zeros((1, 256, 64)), "uint8"), Tensor(np.full((1, 256, 2, 2), 127), "float8_e8m0fnu"))
    with pytest.raises(ValueError, match="l1_weights:.*got format=2"):
        api.transform_weights_for_mega_moe(pair, pair)


def test_store_group_public_configuration_and_prepare_without_process_group(backend, monkeypatch):
    from deep_ep._store import StoreGroup

    stages = []
    group = StoreGroup(None, 0, 2, prefix="api-test")

    def gather(value, stage):
        stages.append(stage)
        return [value, value]

    group.all_gather = gather

    def forbidden(*args, **kwargs):
        pytest.fail("Store API accessed a ProcessGroup")

    monkeypatch.setattr(backend.dist, "get_world_size", forbidden)
    monkeypatch.setattr(backend.dist, "all_gather_object", forbidden)
    buffer = api.SymmBuffer(group, 4, 17, 2, 1024, 256)
    buffer.topk_idx.data[:] = [0, 1]
    buffer.topk_weights.data[:] = [0.4, 0.6]
    buffer.validate_inputs(17)
    buffer.prepare(17)
    buffer.destroy()
    assert stages == [
        "check:configuration",
        "check:input allocation",
        "check:input preflight",
        "check:prepare",
        "check:prepare BS=17",
    ]


def test_non_finite_route_weights_and_bad_capacity_fail_before_launch(backend):
    backend.buffer.topk_weights.data[2, 0] = np.nan
    with pytest.raises(ValueError, match="token=2.*finite"):
        backend.buffer.validate_inputs(17)
    with pytest.raises(ValueError, match="prepare.*BS"):
        backend.buffer.prepare(18)
    assert not any(isinstance(event, tuple) and event[0] == "launch" for event in backend.events)


def test_plan_allocation_failure_is_collective_and_retryable(backend):
    plan_type = backend.ext.Plan

    def fail(*args):
        raise RuntimeError("workspace allocation failed")

    backend.ext.Plan = fail
    with pytest.raises(ValueError, match="rank=0: workspace allocation failed"):
        backend.buffer.prepare(17)
    assert backend.buffer._plan is None and backend.buffer._plan_bs is None
    backend.ext.Plan = plan_type
    backend.buffer.prepare(17)
    assert backend.buffer._plan_bs == 17


def hierarchical_buffer(backend, *, size=32, mma_type="fp8xfp4", capacity=257, **options):
    from deep_ep._store import StoreGroup

    group = StoreGroup(None, 0, size, prefix="hierarchical-api-test")
    group.all_gather = lambda value, stage: [value] * size
    buffer = api.SymmBuffer(
        group,
        size * 2,
        capacity,
        2,
        1024,
        256,
        ranks_per_node=16,
        mma_type=mma_type,
        **options,
    )
    buffer.topk_idx.data[:] = [0, size]
    buffer.topk_weights.data[:] = [0.4, 0.6]
    return buffer


@pytest.mark.parametrize("mma_type", ["fp8xfp4"])
def test_hierarchical_sequence_survives_plan_recreation_and_old_epoch_wrap(backend, mma_type):
    b = hierarchical_buffer(backend, mma_type=mma_type)
    assert b._runtime.ranks_per_node == 16 and b._hierarchical_roce
    pairs = weights() if mma_type == "fp8xfp4" else fp8_weights()
    try:
        for sequence in range(1, 260):
            bs = (1, 31, 32, 33, 138, 144, 255, 256)[sequence % 8]
            api.mega_moe(
                (b.x[:bs], b.x_sf[:bs]),
                b.topk_idx[:bs],
                b.topk_weights[:bs],
                *pairs,
                b,
                out=Tensor(np.zeros((bs, 1024)), "bfloat16"),
            )
            launch = [e[1] for e in backend.events if isinstance(e, tuple) and e[0] == "launch"][-1]
            assert launch[-2] == sequence and b._sequence == sequence
            assert b._plan_bs == bs
        with pytest.raises(ValueError, match=r"BS in \[1,256\]"):
            b.prepare(257)
        assert b._sequence == 259
    finally:
        b.destroy()


def test_hierarchical_old_wheel_retains_bs32_limit(backend):
    del backend.ext.hierarchical_max_tokens
    b = hierarchical_buffer(backend)
    try:
        b.prepare(32)
        with pytest.raises(ValueError, match=r"BS in \[1,32\]"):
            b.prepare(33)
        assert b._plan_bs == 32
    finally:
        b.destroy()


def test_hierarchical_launch_failure_and_sequence_exhaustion(backend, monkeypatch):
    b = hierarchical_buffer(backend)
    try:
        b.prepare(1)
        plan = b._plan
        launch = plan.forward

        def fail(*args):
            raise RuntimeError("launch rejected")

        monkeypatch.setattr(plan, "forward", fail)
        with pytest.raises(RuntimeError, match="launch rejected"):
            api.fp8_fp4_mega_moe(Tensor(np.zeros((1, 1024)), "bfloat16"), *weights(), b)
        assert b._sequence == 0
        monkeypatch.setattr(plan, "forward", launch)
        b._sequence = 0xFFFFFFFE
        api.fp8_fp4_mega_moe(Tensor(np.zeros((1, 1024)), "bfloat16"), *weights(), b)
        assert b._sequence == 0xFFFFFFFF
        count = len(backend.events)
        with pytest.raises(RuntimeError, match="sequence exhausted"):
            api.fp8_fp4_mega_moe(Tensor(np.zeros((1, 1024)), "bfloat16"), *weights(), b)
        assert not any(isinstance(e, tuple) and e[0] == "launch" for e in backend.events[count:])
    finally:
        b.destroy()


@pytest.mark.parametrize("size,mma_type", [(64, "fp8xfp4")])
def test_hierarchical_unsupported_configuration_fails_collectively(backend, size, mma_type):
    with pytest.raises(ValueError, match="MTE\\+RoCE MegaMoE requires EP32"):
        hierarchical_buffer(backend, size=size, mma_type=mma_type)


def test_sixteen_rank_single_host_keeps_mte(backend):
    b = hierarchical_buffer(backend, size=16)
    try:
        assert not b._hierarchical_roce
        b.prepare(32)
        api.fp8_fp4_mega_moe(Tensor(np.zeros((32, 1024)), "bfloat16"), *weights(), b)
        launch = [e[1] for e in backend.events if isinstance(e, tuple) and e[0] == "launch"][-1]
        assert launch[-2] == 0 and b._sequence == 0
    finally:
        b.destroy()


@pytest.mark.parametrize("protocol", list(range(5, 18)))
def test_extension_rejects_native_without_expanded_hierarchical_support(monkeypatch, protocol):
    api._extension.cache_clear()
    core = SimpleNamespace(create_megamoe_runtime=lambda: None)
    extension = SimpleNamespace(api_protocol=protocol)
    monkeypatch.setattr(api, "require_native", lambda _: core)
    monkeypatch.setattr(api.importlib, "import_module", lambda _: extension)
    try:
        if protocol != 17:
            with pytest.raises(RuntimeError, match="versions differ"):
                api._extension()
        else:
            assert api._extension() is extension
    finally:
        api._extension.cache_clear()


def test_hierarchical_plan_has_no_selectable_algorithm_arguments(backend):
    sizes = []
    backend.ext.buffer_size = lambda *args: sizes.append(args) or (2 << 20)
    b = hierarchical_buffer(backend)
    try:
        for bs in (1, 32, 256):
            b.prepare(bs)
            assert len(b._plan.options) == 14
            assert b._plan.options[-5:] == (4096, 2 << 20, (2, 257, 2, 1024), 0, 1)
        assert sizes == [(32, 2, 257, 2, 1024, 16, 1)]
        assert b._runtime.ranks_per_node == 16
    finally:
        b.destroy()


@pytest.mark.parametrize(
    "name",
    [
        "combine_reduce_qps",
        "dispatch_workers",
        "reduce_lane_init",
        "combine_sparse_scan",
        "reduce_double_buffer",
        "combine_single_row",
        "combine_ready_trace",
        "dispatch_skip_empty",
        "dispatch_lookahead",
        "dispatch_single_row",
        "combine_cross_expert",
    ],
)
def test_retired_algorithm_switches_are_not_public_api(backend, monkeypatch, name):
    monkeypatch.setattr(api, "_MegaRuntime", lambda *a, **kw: pytest.fail("Retired option started SHMEM"))
    with pytest.raises(TypeError, match="unexpected keyword"):
        hierarchical_buffer(backend, **{name: True})


@pytest.mark.parametrize("case", ["fp8", "old-core"])
def test_qp8_rejects_incompatible_mode_or_old_runtime_before_shmem(backend, monkeypatch, case):
    monkeypatch.setattr(api, "_MegaRuntime", lambda *a, **kw: pytest.fail("Invalid configuration started SHMEM"))
    if case == "old-core":
        monkeypatch.setattr(api, "require_native", lambda _: SimpleNamespace(megamoe_combine_reduce_version=1))
    with pytest.raises(ValueError, match="FP8xFP4|Rebuild"):
        hierarchical_buffer(backend, mma_type="fp8xfp8" if case == "fp8" else "fp8xfp4")


@pytest.mark.parametrize("clamp", [None, 0, 2, 5, 10, 1 / 3])
def test_public_clamp_is_a_captured_fp32_scalar(backend, clamp):
    b = backend.buffer
    y = Tensor(np.zeros((17, 1024)), "bfloat16")
    api.fp8_fp4_mega_moe(y, *weights(), b, activation_clamp=clamp)
    args = next(item[1] for item in backend.events if isinstance(item, tuple) and item[0] == "launch")
    assert args[-3] == (np.finfo(np.float32).max if clamp is None else float(np.float32(clamp)))


@pytest.mark.parametrize("bad", [True, -1, float("inf"), float("nan"), 2**129, Tensor([1], "float32")])
def test_clamp_rejects_invalid_or_device_values(bad):
    with pytest.raises((ValueError, TypeError)):
        api._activation_clamp(bad)


def test_views_cache_separate_domains_freeze_and_close_independently(backend):
    root = hierarchical_buffer(backend, capacity=64, num_max_configs=3)
    child = api.SymmBuffer(root.group, 64, 32, 2, 1024, 256, ranks_per_node=16, base=root)
    root.prepare(32)
    child.prepare(32)
    child.prepare(16)
    plans = [root._plans[32], child._plans[32], child._plans[16]]
    assert [p.options[-2] for p in plans] == [0, 1, 2]
    api.fp8_fp4_mega_moe(Tensor(np.zeros((32, 1024)), "bfloat16"), *weights(), root)
    assert root._domains.frozen and child._domains is root._domains
    root.destroy()
    assert not child._runtime.closed
    child.prepare(32)
    assert child._plan is plans[1]
    with pytest.raises(ValueError, match="frozen"):
        child.prepare(31)
    assert child._domains.failed
    child.destroy()
    assert child._runtime.closed


def test_list_replacement_preserves_both_captured_tables_and_unchanged_experts(backend):
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    def raw_groups():
        return [[(w[i].clone(), s[i].clone()) for i in range(2)] for w, s in weights()]

    l1, l2 = raw_groups()
    owner = prepare_weights_for_mega_moe(l1, l2)
    tables = [owner.select(f)[2] for f in (True, False)]
    old = [t.data.copy() for t in tables]
    l1, l2 = raw_groups()
    owner.update_experts([1], l1[1:], l2[1:], replace=True)
    for fused, table, previous in zip((True, False), tables, old):
        assert table is owner.select(fused)[2]
        assert np.array_equal(table.data[:, 0], previous[:, 0])
        assert np.all(table.data[:, 1] != previous[:, 1])
    api.fp8_fp4_mega_moe(Tensor(np.zeros((17, 1024)), "bfloat16"), owner.l1, owner.l2, backend.buffer)
    assert owner in backend.buffer._owners
    assert owner.check_versions()
    owner.release_retired()


def test_list_failed_table_copy_requires_full_recovery(backend, monkeypatch):
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    groups = [[(w[i].clone(), s[i].clone()) for i in range(2)] for w, s in weights()]
    owner = prepare_weights_for_mega_moe(*groups)
    table = owner.select(False)[2]
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise RuntimeError("table copy failed")

        patch.setattr(table, "copy_", fail)
        with pytest.raises(RuntimeError, match="copy failed"):
            owner.update_experts([1], *(pairs[1:] for pairs in groups), replace=True)
    with pytest.raises(RuntimeError, match="incomplete"):
        owner.select(True)
    with pytest.raises(RuntimeError, match="retry all"):
        owner.update_experts([0], *(pairs[:1] for pairs in groups))
    owner.update_experts([1], *(pairs[1:] for pairs in groups))
    assert owner.select(False)[2] is table
    assert table.data[0, 1] == owner.expert_views(1)["planar_l1"][0].data_ptr()


def test_external_eplb_refresh_updates_both_layouts_in_place(backend):
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    owner = prepare_weights_for_mega_moe(*weights(), input_layout="planar_nz")
    views = owner.expert_views(1)
    planar, fused = views["planar_l1"], views["l1"]
    addresses = [t.data_ptr() for pair in (planar, fused) for t in pair]
    for target, inverse in (("planar", False), ("fused", True)):
        src, dst = (fused, planar) if inverse else (planar, fused)
        for t in src:
            for block, values in enumerate(np.split(t.data, 4)):
                values.fill(8 + 2 * block + inverse)
        owner.refresh_gmm1(target, [1])
        for source, result in zip(src, dst):
            expected = np.concatenate([np.split(source.data, 4)[i] for i in (0, 2, 1, 3)])
            np.testing.assert_array_equal(result.data, expected)
        assert addresses == [t.data_ptr() for pair in (planar, fused) for t in pair]
        assert owner.check_versions()
