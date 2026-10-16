// Copyright (c) 2026, Lu Lu

#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include "kernel_operator.h"

constexpr int ACLSHMEM_CMP_EQ = 0;

namespace RoceTest {
constexpr uint32_t kPeer = 21U;
constexpr uint32_t kCommitOffset = 1536U;
inline std::array<uint8_t, 4096> local{};
inline std::array<uint8_t, 4096> remote{};
inline std::array<uint8_t, 256> scratch{};
struct Write {
    uint32_t destination;
    uint8_t* source;
    uint32_t bytes;
};
inline std::vector<Write> queued;
inline uint32_t writes = 0;
inline uint32_t quiets = 0;

inline void CheckResource(uint32_t peer, uint32_t qp, uint8_t* ub, uint32_t event)
{
    if (peer != kPeer || qp != 1U || ub != scratch.data() || event != 7U) {
        throw std::runtime_error("transport used another owner's peer/QP/UB/event");
    }
}

inline uint32_t CheckSymmetric(uint8_t* address, uint32_t bytes)
{
    const auto offset = reinterpret_cast<uintptr_t>(address) - reinterpret_cast<uintptr_t>(local.data());
    if (offset > local.size() || bytes > local.size() - offset) {
        throw std::runtime_error("operand was not a local symmetric address");
    }
    return static_cast<uint32_t>(offset);
}
}  // namespace RoceTest

#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
template <typename T>
void aclshmemx_roce_qp_put_nbi(T* destination, T* source, T* scratch, uint32_t count, int peer, uint32_t qp,
                               uint32_t event)
{
    static_assert(sizeof(T) == 1U);
    using namespace RoceTest;
    CheckResource(static_cast<uint32_t>(peer), qp, scratch, event);
    const uint32_t offset = CheckSymmetric(destination, count);
    CheckSymmetric(source, count);
    if (offset == kCommitOffset && !queued.empty()) {
        throw std::runtime_error("commit can overtake outstanding data writes");
    }
    queued.push_back({offset, source, count});
    ++writes;
    operations.push_back(1);
}

template <typename T>
void aclshmemx_roce_qp_quiet(uint32_t peer, uint32_t qp, T* scratch, uint32_t event)
{
    static_assert(sizeof(T) == 1U);
    using namespace RoceTest;
    CheckResource(peer, qp, scratch, event);
    // Delayed completion and reversed WQE execution expose premature signaling.
    for (auto item = queued.rbegin(); item != queued.rend(); ++item) {
        std::copy_n(item->source, item->bytes, remote.data() + item->destination);
    }
    queued.clear();
    ++quiets;
    operations.push_back(2);
}
#endif

inline int32_t aclshmem_signal_wait_until(int32_t* address, int comparison, int32_t value)
{
    int32_t observed;
    std::memcpy(&observed, address, sizeof(observed));
    if (comparison != ACLSHMEM_CMP_EQ || observed != value) {
        throw std::runtime_error("receiver did not wait for its exact invocation");
    }
    RoceTest::signalObserved = true;
    return observed;
}
