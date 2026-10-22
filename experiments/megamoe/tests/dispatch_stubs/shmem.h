// Copyright (c) 2026, Lu Lu
#pragma once

// Asynchronous transport double for the real Dispatch and RoCE helpers. The
// symmetric heap uses the production layout; this does not simulate NPU caches.
#include <algorithm>
#include <cstring>
#include <vector>
#include "kernel_operator.h"

constexpr int ACLSHMEM_CMP_EQ = 0;

namespace DispatchTest {
struct Write {
    uint32_t destination;
    uint32_t source;
    uint32_t bytes;
    std::vector<uint8_t> sourceAtPost;
};
inline std::vector<uint8_t> local;
inline std::vector<uint8_t> remote;
inline std::vector<Write> posts;
inline std::vector<Write> queued;
inline uint32_t peer = 0U;
inline uint32_t metaCommit = 0U;
inline uint32_t payloadCommit = 0U;
inline uint32_t metaReady = 0U;
inline uint32_t relayReady = 0U;
inline uint32_t txDone = 0U;
inline uint32_t sequence = 0U;
inline uint32_t waits = 0U;
inline uint32_t quiets = 0U;
inline uint32_t packCalls = 0U;
inline uint32_t relayCalls = 0U;
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
    Check(RoceTest::blockIndex == 1 && RoceTest::subBlockIndex == 1, "QP owner is not physical AIV1");
    Check(actualPeer == peer && qp == 1U && reinterpret_cast<uintptr_t>(ub) == kScratchOffset && event == 7U,
          "Dispatch used an unowned peer/QP/UB/event");
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
        Check(count == sizeof(uint32_t) && queued.empty(), "commit overtook unfinished payload");
    }
    Write operation{destinationOffset, sourceOffset, count, std::vector<uint8_t>(source, source + count)};
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
    // Execute in reverse order to expose a missing payload-before-commit quiet.
    // Bytes are read at completion, so reusing the GM source early is observable.
    for (auto item = queued.rbegin(); item != queued.rend(); ++item) {
        Check(std::equal(item->sourceAtPost.begin(), item->sourceAtPost.end(), local.data() + item->source),
              "in-flight GM source was reused before quiet");
        std::copy_n(local.data() + item->source, item->bytes, remote.data() + item->destination);
    }
    queued.clear();
    ++quiets;
    RoceTest::operations.push_back(2);
}

inline int32_t aclshmem_signal_wait_until(int32_t* address, int comparison, int32_t value)
{
    using namespace DispatchTest;
    Check(Offset(reinterpret_cast<uint8_t*>(address), sizeof(int32_t)) == payloadCommit,
          "receiver did not wait on the unified Dispatch commit");
    Check(comparison == ACLSHMEM_CMP_EQ && static_cast<uint32_t>(value) == sequence &&
              ReadWord(local, payloadCommit) == sequence,
          "receiver accepted a stale invocation");
    if (RoceTest::blockIndex == 1) {
        Check(queued.empty() && ReadWord(remote, payloadCommit) == sequence && ReadWord(local, txDone) == sequence,
              "sender's incoming wait preceded outgoing completion and TxDone");
    }
    ++waits;
    RoceTest::signalObserved = true;
    return value;
}
