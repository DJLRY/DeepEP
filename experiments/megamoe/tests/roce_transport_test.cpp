// Copyright (c) 2026, Lu Lu

#include <cassert>
#include "../csrc/kernels/mega_moe/common/mega_moe_roce.hpp"

int main()
{
    using namespace MegaMoeImpl;
    using namespace RoceTest;
    static_assert(MEGAMOE_ROCE_SCRATCH_BYTES == scratch.size());
    constexpr uint32_t sourceOffset = 2048U;
    constexpr uint32_t sequenceOffset = 3072U;
    constexpr uint32_t sequence = 130U;  // Past the old route epoch's 7-bit wrap.

#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    remote.fill(0xA5U);
    for (uint32_t i = 0; i < 96U; ++i) {
        local[sourceOffset + i] = static_cast<uint8_t>(i);
    }
    std::memcpy(local.data() + sequenceOffset, &sequence, sizeof(sequence));
    RocePutBytes(local.data(), local.data() + sourceOffset, 32U, kPeer, scratch.data());
    RocePutBytes(local.data() + 64U, local.data() + sourceOffset + 64U, 32U, kPeer, scratch.data());
    assert(remote[0] == 0xA5U);  // Put is asynchronous.
    scratch.fill(0xCCU);         // In-flight operands are GM; SDK scratch can be reused.
    RocePutBytes(nullptr, nullptr, 0U, kPeer, scratch.data());
    assert(writes == 2U);
    RocePublishCommit(local.data() + kCommitOffset, local.data() + sequenceOffset, kPeer, scratch.data());
    assert(queued.empty());
    assert(writes == 3U && quiets == 2U);
    assert((operations == std::vector<int>{1, 1, 2, 1, 2}));
    for (uint32_t i = 0; i < 96U; ++i) {
        assert(remote[i] == (i >= 32U && i < 64U ? 0xA5U : static_cast<uint8_t>(i)));
    }
    local[sourceOffset] = 0xEEU;  // The source can be reused after completion.
    assert(remote[0] == 0U);
    assert(!mteCanRead);
    RoceWaitCommit(reinterpret_cast<int32_t*>(remote.data() + kCommitOffset), sequence);
    assert(mteCanRead && !scalarEventSet);

    // Zero payload still commits a fresh epoch; receivers cannot reuse old flags.
    const uint32_t nextSequence = sequence + 1U;
    std::memcpy(local.data() + sequenceOffset, &nextSequence, sizeof(nextSequence));
    RocePublishCommit(local.data() + kCommitOffset, local.data() + sequenceOffset, kPeer, scratch.data());
    int32_t observed;
    std::memcpy(&observed, remote.data() + kCommitOffset, sizeof(observed));
    assert(observed == static_cast<int32_t>(nextSequence));
    assert(writes == 4U && quiets == 4U);
#else
    bool trapped = false;
    try {
        RocePutBytes(local.data(), local.data() + sourceOffset, 32U, kPeer, scratch.data());
    } catch (const std::runtime_error&) {
        trapped = true;
    }
    assert(trapped);
    trapped = false;
    try {
        RoceQuiet(kPeer, scratch.data());
    } catch (const std::runtime_error&) {
        trapped = true;
    }
    assert(trapped);
    (void)sequenceOffset;
    (void)sequence;
#endif
    return 0;
}
