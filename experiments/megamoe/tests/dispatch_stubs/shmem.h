// Copyright (c) 2026, Lu Lu
#pragma once

// Asynchronous transport double for the real Dispatch and RoCE helpers. The
// symmetric heap uses the production layout; this does not simulate NPU caches.
#include <algorithm>
#include <cstring>
#include <functional>
#include <vector>
#include "kernel_operator.h"

#define MEGA_MOE_ROCE_PROGRESS_H

constexpr int ACLSHMEM_CMP_EQ = 0;

namespace DispatchTest {
struct Write {
    uint32_t destination;
    uint32_t source;
    uint32_t bytes;
    uint32_t qp;
    std::vector<uint8_t> sourceAtPost;
    bool delivered = false;
};
inline std::vector<uint8_t> local;
inline std::vector<uint8_t> remote;
inline std::vector<Write> posts;
inline std::vector<Write> queued;
inline uint32_t peer = 0U;
inline uint32_t metaCommit = 0U;
inline uint32_t payloadCommit = 0U;
inline uint32_t metaReady = 0U;
inline uint32_t sequence = 0U;
inline uint32_t waits = 0U;
inline uint32_t quiets = 0U;
inline uint32_t packCalls = 0U;
inline uint32_t relayCalls = 0U;
inline uint32_t receiveLeaderJob = 0U;
inline uint32_t progressCalls = 0U;
inline std::function<void()> completePacking;
constexpr uint32_t kScratchOffset = 4096U;

inline void Check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

inline uint32_t Offset(const uint8_t* address, uint32_t bytes)
{
    const auto offset = reinterpret_cast<uintptr_t>(address) - reinterpret_cast<uintptr_t>(local.data());
    Check(offset <= local.size() && bytes <= local.size() - offset, "nonlocal or out-of-bounds symmetric operand");
    return static_cast<uint32_t>(offset);
}

inline uint32_t ReadWord(const std::vector<uint8_t>& heap, uint32_t offset)
{
    uint32_t value;
    std::memcpy(&value, heap.data() + offset, sizeof(value));
    return value;
}

inline void CheckResource(uint32_t actualPeer, uint32_t qp, uint8_t* ub, uint32_t event)
{
    const uint32_t job = static_cast<uint32_t>(RoceTest::blockIndex);
    Check((job == 0U && qp == 9U) || (job < 16U && (job & 1U) != 0U && qp == job / 2U + 1U),
          "QP was submitted or polled by another owner");
    Check(actualPeer == peer && reinterpret_cast<uintptr_t>(ub) == kScratchOffset && event == 7U,
          "Dispatch used an unowned peer/QP/UB/event");
}

inline void Deliver(Write& item)
{
    Check(std::equal(item.sourceAtPost.begin(), item.sourceAtPost.end(), local.data() + item.source),
          "in-flight GM source was reused before quiet");
    if (!item.delivered) {
        std::copy_n(local.data() + item.source, item.bytes, remote.data() + item.destination);
        item.delivered = true;
    }
}

inline bool IsCommit(const Write& item)
{
    return item.destination == metaCommit ||
           (item.destination >= payloadCommit && item.destination < payloadCommit + 8U * 512U &&
            (item.destination - payloadCommit) % 512U == 0U);
}

// Each queue can deliver independently. A metadata commit must never complete
// any payload queue; within one QP only its commit orders earlier writes.
inline void DeliverPosted(uint32_t qp)
{
    std::vector<size_t> pending;
    for (size_t i = 0U; i < queued.size(); ++i) {
        if (queued[i].qp != qp) continue;
        if (IsCommit(queued[i])) {
            for (auto it = pending.rbegin(); it != pending.rend(); ++it) Deliver(queued[*it]);
            pending.clear();
            Deliver(queued[i]);
        } else {
            pending.push_back(i);
        }
    }
    for (auto it = pending.rbegin(); it != pending.rend(); ++it) Deliver(queued[*it]);
}
}  // namespace DispatchTest

template <typename T>
void aclshmemx_roce_qp_put_nbi(T* destination, T* source, T* scratch, uint32_t count, int peer, uint32_t qp,
                               uint32_t event)
{
    static_assert(sizeof(T) == 1U);
    using namespace DispatchTest;
    CheckResource(static_cast<uint32_t>(peer), qp, scratch, event);
    const uint32_t destinationOffset = Offset(destination, count);
    const uint32_t sourceOffset = Offset(source, count);
    Write operation{destinationOffset, sourceOffset, count, qp, std::vector<uint8_t>(source, source + count)};
    if (IsCommit(operation)) {
        Check(count == sizeof(uint32_t), "commit was not a single-fragment epoch");
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
        Check(std::none_of(queued.begin(), queued.end(), [qp](const auto& w) { return w.qp == qp; }),
              "XSCALE commit overtook unfinished writes on its QP");
#endif
    }
    posts.push_back(operation);
    queued.push_back(operation);
    RoceTest::operations.push_back(1);
}

template <typename T>
void aclshmemx_roce_qp_quiet(uint32_t peer, uint32_t qp, T* scratch, uint32_t event)
{
    static_assert(sizeof(T) == 1U);
    using namespace DispatchTest;
    CheckResource(peer, qp, scratch, event);
    DeliverPosted(qp);
    queued.erase(std::remove_if(queued.begin(), queued.end(), [qp](const auto& w) { return w.qp == qp; }),
                 queued.end());
    ++quiets;
    RoceTest::operations.push_back(2);
}

namespace MegaMoeImpl {
inline uint32_t RoceProgressReady(uint32_t peer, uint32_t qp, uint8_t* scratch, uint32_t event)
{
    DispatchTest::CheckResource(peer, qp, scratch, event);
    ++DispatchTest::progressCalls;
    return 0U;
}
}  // namespace MegaMoeImpl

inline int aclshmem_int32_test(int32_t* address, int comparison, int32_t value)
{
    using namespace DispatchTest;
    if (completePacking) {
        auto pack = std::move(completePacking);
        completePacking = nullptr;
        pack();
    }
    const auto offset = Offset(reinterpret_cast<uint8_t*>(address), sizeof(int32_t));
    Check(comparison == ACLSHMEM_CMP_EQ && static_cast<uint32_t>(value) == sequence, "incorrect invocation wait");
    if (ReadWord(local, offset) != sequence) return 0;
    if (offset == metaCommit) ++waits;
    RoceTest::signalObserved = true;
    return 1;
}

inline int32_t aclshmem_signal_wait_until(int32_t* address, int comparison, int32_t value)
{
    DispatchTest::Check(
        DispatchTest::Offset(reinterpret_cast<uint8_t*>(address), sizeof(int32_t)) == DispatchTest::metaCommit,
        "receiver did not wait on the metadata Dispatch commit");
    DispatchTest::Check(aclshmem_int32_test(address, comparison, value) != 0, "receiver accepted a stale invocation");
    return value;
}
