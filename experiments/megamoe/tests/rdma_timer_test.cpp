// Copyright (c) 2026, Lu Lu

// Host doubles verify timer state and ordering, not device/NIC visibility.
#include <cassert>
#include "../csrc/kernels/mega_moe/common/mega_moe_roce.hpp"

int main()
{
    using namespace AscendTimer;
    using namespace MegaMoeImpl;
    using namespace RoceTest;
    std::vector<int64_t> buffer(TOTAL_BUFFER_SIZE, 0);
    AscendTimerDevice timer;
    timer.Init(reinterpret_cast<uint8_t*>(buffer.data()));
    constexpr int core = 1;  // group 0 / AIV0
    auto logical = [](int type, int slot) { return FIXED_TIMING_COUNT + type * MAX_DYNAMIC_ITER + slot; };
    auto start = [&](int type, int slot) {
        return buffer[core * N_TIMING_COUNTER_PER_CORE_ALIGN + logical(type, slot) * 2];
    };
    auto end = [&](int type, int slot) {
        return buffer[core * N_TIMING_COUNTER_PER_CORE_ALIGN + logical(type, slot) * 2 + 1];
    };
    auto count = [&](int type) { return buffer[N_TIMING_COUNTER + core * DYNAMIC_ITER_PER_CORE_ALIGN + type]; };
    auto meta = [&](int type, int slot, int word) {
        return buffer[TASK_METADATA_OFFSET + ((core * DYNAMIC_TYPE_COUNT + type) * MAX_DYNAMIC_ITER + slot) * 2 + word];
    };
    auto wave = [&](int type, int slot) {
        return buffer[WAVE_ID_OFFSET + core * WAVE_ID_PER_CORE_ALIGN + logical(type, slot)];
    };

    // Exercise all three phases while an outer compute tile and dependency wait
    // are live. The transport must not overwrite either timer handle/metadata.
    timer.SetWave(7);
    timer.SetSlice(3, 256, 32);
    for (uint32_t phase = 0; phase < 3; ++phase) {
        auto stage = static_cast<RdmaPhase>(phase);
        timer.StartTile(AIV_COMBINE, 0, static_cast<uint32_t>(-1), 32);
        timer.BeginWait(AIV_COMBINE_WAIT);
        auto barriers = pipelineBarriers;
        uint32_t sequence = 130 + phase;
        std::memcpy(local.data() + 3072, &sequence, sizeof(sequence));
        operations.clear();
        RocePutBytes(nullptr, nullptr, 0, kPeer, scratch.data(), &timer, stage);
        RocePutBytes(local.data(), local.data() + 2048, 32, kPeer, scratch.data(), &timer, stage);
        RocePublishCommit(local.data() + kCommitOffset, local.data() + 3072, kPeer, scratch.data(), &timer, stage);
        RoceWaitCommit(reinterpret_cast<int32_t*>(remote.data() + kCommitOffset), sequence, &timer, stage, kPeer);
        assert((operations == std::vector<int>{1, 2, 1, 2}));
        assert(pipelineBarriers == barriers);  // No timer-induced pipeline drain.
        timer.EndWait();
        timer.EndTile();
    }
    timer.FlushDynamicCounts();
#ifdef DEEPEP_MEGAMOE_TIMER
    assert(count(AIV_RDMA_POST) == 6 && count(AIV_RDMA_QUIET_WAIT) == 6 && count(AIV_RDMA_RECV_WAIT) == 3);
    assert(count(AIV_COMBINE) == 3 && count(AIV_COMBINE_WAIT) == 3);
    for (int phase = 0; phase < 3; ++phase) {
        assert(end(AIV_COMBINE, phase) > start(AIV_COMBINE, phase));
        assert(end(AIV_COMBINE_WAIT, phase) <= start(AIV_COMBINE, phase));
        assert(meta(AIV_COMBINE, phase, 0) == (int64_t(4) << 32 | 256));
        assert(wave(AIV_COMBINE, phase) == 7);
        for (int type : {AIV_RDMA_POST, AIV_RDMA_QUIET_WAIT, AIV_RDMA_RECV_WAIT}) {
            int slot = type == AIV_RDMA_RECV_WAIT ? phase : phase * 2;
            assert(end(type, slot) > start(type, slot));
            assert(meta(type, slot, 0) == (int64_t(kPeer + 1) << 32 | (phase + 1)));
            assert(wave(type, slot) == NO_WAVE);
        }
        assert(meta(AIV_RDMA_POST, phase * 2, 1) == 32);
        assert(meta(AIV_RDMA_POST, phase * 2 + 1, 1) == (int64_t(1) << 32 | 4));
        assert(meta(AIV_RDMA_QUIET_WAIT, phase * 2, 1) == 0);
        assert(meta(AIV_RDMA_RECV_WAIT, phase, 1) == (int64_t(1) << 32 | 4));
    }
    // Span state survives independent RDMA handles too, with no extra split.
    timer.StartSpan(AIV_DISPATCH);
    {
        AscendTimerRdmaScope rdma(&timer, AIV_RDMA_POST, RdmaPhase::Combine, kPeer, 8, false);
    }
    timer.EndSpan(AIV_DISPATCH);
    timer.FlushDynamicCounts();
    assert(count(AIV_DISPATCH) == 1 && end(AIV_DISPATCH, 0) > start(AIV_DISPATCH, 0));
    assert(wave(AIV_DISPATCH, 0) == 7);
    // Overflow must retain the real count for host rejection, without corrupting
    // the next family's time slots or the currently active compute span.
    auto neighbor = start(AIV_RDMA_QUIET_WAIT, 0);
    for (int i = 7; i <= MAX_DYNAMIC_ITER; ++i) {
        AscendTimerRdmaScope rdma(&timer, AIV_RDMA_POST, RdmaPhase::Combine, kPeer, 8, false);
    }
    timer.FlushDynamicCounts();
    assert(count(AIV_RDMA_POST) == MAX_DYNAMIC_ITER + 1);
    assert(start(AIV_RDMA_QUIET_WAIT, 0) == neighbor);
#else
    for (auto value : buffer) {
        assert(value == 0);
    }
    assert(!timer.Enabled() && pipelineBarriers == 0);
    (void)start;
    (void)end;
    (void)count;
    (void)meta;
    (void)wave;
#endif
    AscendTimerDevice disabled;
    disabled.Init(nullptr);
    assert(!disabled.Enabled());
    assert(disabled.BeginRdma(AIV_RDMA_POST, RdmaPhase::Combine, kPeer, 4, true) == -1);
    disabled.EndRdma(AIV_RDMA_POST, -1);
    return 0;
}
