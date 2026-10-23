// Copyright (c) 2026, Lu Lu
#pragma once

// Independent asynchronous QPs for the production Dispatch protocol. Completions
// deliberately read GM late and may reorder data; this does not model NIC caches.
#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <vector>
#include "kernel_operator.h"

constexpr int ACLSHMEM_CMP_EQ = 0;

namespace DispatchTest {
struct Write {
    uint32_t qp;
    uint32_t destination;
    uint32_t source;
    uint32_t bytes;
    std::vector<uint8_t> sourceAtPost;
};
inline std::vector<uint8_t> local, remote;
inline std::vector<Write> posts, queued;
inline uint32_t peer = 0U, metaCommit = 0U, payloadCommit = 0U;
inline uint32_t metaReady = 0U, relayReady = 0U, txDone = 0U, sequence = 0U;
inline uint32_t waits = 0U, quiets = 0U, packCalls = 0U, relayCalls = 0U;
inline uint32_t receiveLeaderJob = 0U, metaSenderJob = 0U;
inline std::array<uint32_t, 128> metaAcquired{}, payloadAcquired{};
inline std::function<void()> onPayloadWait;
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

inline void SetWord(std::vector<uint8_t>& heap, uint32_t offset, uint32_t value)
{
    std::memcpy(heap.data() + offset, &value, sizeof(value));
}

inline size_t Pending(uint32_t qp)
{
    return static_cast<size_t>(
        std::count_if(queued.begin(), queued.end(), [qp](const Write& write) { return write.qp == qp; }));
}

inline void CheckResource(uint32_t actualPeer, uint32_t qp, uint8_t* ub, uint32_t event)
{
    const uint32_t owner = qp == 1U ? 1U : metaSenderJob;
    Check((qp == 1U || qp == 2U) && RoceTest::blockIndex == static_cast<int>(owner) && RoceTest::subBlockIndex == 1,
          "QP is not owned by its designated physical AIV1");
    Check(actualPeer == peer && reinterpret_cast<uintptr_t>(ub) == kScratchOffset && event == 7U,
          "Dispatch used an unowned peer/UB/event");
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
    if (destinationOffset == metaCommit || destinationOffset == payloadCommit) {
        Check(count == sizeof(uint32_t) && Pending(qp) == 0U, "commit overtook unfinished data on its QP");
        Check(qp == (destinationOffset == metaCommit ? 2U : 1U), "commit used the other channel's QP");
    }
    Write operation{qp, destinationOffset, sourceOffset, count, std::vector<uint8_t>(source, source + count)};
    posts.push_back(operation);
    queued.push_back(operation);
    RoceTest::operations.push_back(static_cast<int>(qp * 10U + 1U));
}

template <typename T>
void aclshmemx_roce_qp_quiet(uint32_t peer, uint32_t qp, T* scratch, uint32_t event)
{
    static_assert(sizeof(T) == 1U);
    using namespace DispatchTest;
    CheckResource(peer, qp, scratch, event);
    // A quiet drains only its owner's QP, including neither the other data
    // transfer nor its commit. Reverse completion exposes missing first quiets.
    for (auto item = queued.rbegin(); item != queued.rend(); ++item) {
        if (item->qp != qp) continue;
        Check(std::equal(item->sourceAtPost.begin(), item->sourceAtPost.end(), local.data() + item->source),
              "in-flight GM source was reused before quiet");
        std::copy_n(local.data() + item->source, item->bytes, remote.data() + item->destination);
    }
    queued.erase(std::remove_if(queued.begin(), queued.end(), [qp](const Write& write) { return write.qp == qp; }),
                 queued.end());
    ++quiets;
    RoceTest::operations.push_back(static_cast<int>(qp * 10U + 2U));
}

inline int32_t aclshmem_signal_wait_until(int32_t* address, int comparison, int32_t value)
{
    using namespace DispatchTest;
    const uint32_t offset = Offset(reinterpret_cast<uint8_t*>(address), sizeof(int32_t));
    Check(offset == metaCommit || offset == payloadCommit, "receiver waited on an unrelated signal");
    Check(RoceTest::blockIndex == static_cast<int>(offset == metaCommit ? receiveLeaderJob : 0U),
          "incoming completion was acquired by the wrong receive leader");
    if (offset == payloadCommit && onPayloadWait) {
        auto progress = std::move(onPayloadWait);
        onPayloadWait = nullptr;
        progress();
    }
    Check(comparison == ACLSHMEM_CMP_EQ && static_cast<uint32_t>(value) == sequence &&
              ReadWord(local, offset) == sequence,
          "receiver accepted a stale invocation");
    (offset == metaCommit ? metaAcquired : payloadAcquired)[RoceTest::blockIndex] = sequence;
    ++waits;
    RoceTest::signalObserved = true;
    return value;
}
