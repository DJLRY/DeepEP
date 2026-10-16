# Copyright (c) 2026, Lu Lu

"""CPU specification tests for the two-host MegaMoE wire contract.

The model represents messages and dependency events, not device instructions,
cache coherence, SHMEM completion, or NPU scheduling. It compares hierarchical
transport with direct routing and uses the existing BF16 oracle for arithmetic.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import groupby
import random

import numpy as np
import pytest

from deep_ep_experimental.megamoe.reference import bf16, quantize_input


WORLD = 32
HOST_SIZE = 16
EXPERTS_PER_RANK = 32


def host(rank):
    return rank // HOST_SIZE


def pair(rank):
    return (1 - host(rank)) * HOST_SIZE + rank % HOST_SIZE


def routing(bs, topk, pattern):
    ids = np.empty((WORLD, bs, topk), dtype=np.int32)
    for source in range(WORLD):
        for token in range(bs):
            for slot in range(topk):
                remote = pattern in ("remote", "duplicate-rank") or (
                    pattern == "mixed" and (slot + token + source) % 3 == 0
                )
                target_host = 1 - host(source) if remote else host(source)
                local_rank = (source + 7) % HOST_SIZE if pattern == "duplicate-rank" else (slot + token) % HOST_SIZE
                ids[source, token, slot] = (target_host * HOST_SIZE + local_rank) * EXPERTS_PER_RANK + slot
    return ids


def routes(ids):
    for source, token, slot in np.ndindex(ids.shape):
        expert = int(ids[source, token, slot])
        yield (source, token, slot), expert // EXPERTS_PER_RANK, expert


def input_records(bs, seed):
    rng = np.random.default_rng(seed)
    data, scales = quantize_input(rng.normal(size=(WORLD, bs, 32)).astype(np.float32))
    return {
        (source, token): data[source, token].tobytes() + scales[source, token].tobytes()
        for source in range(WORLD)
        for token in range(bs)
    }


def hierarchical_dispatch(ids, records):
    """Deduplicate payload messages by original source/token/destination host."""
    transfer_keys = {
        (source, token, host(owner)) for (source, token, _), owner, _ in routes(ids) if host(owner) != host(source)
    }
    relayed = {(pair(source), token): records[source, token] for source, token, _ in transfer_keys}
    consumers = {}
    for key, owner, expert in routes(ids):
        source, token, _ = key
        payload = records[source, token] if host(owner) == host(source) else relayed[pair(source), token]
        consumers[owner, expert, key] = payload
    return transfer_keys, consumers


@dataclass
class PersistentGatewayMemory:
    # Flags deliberately survive calls, including calls with different BS/routes.
    epochs: dict = field(default_factory=dict)
    rows: dict = field(default_factory=dict)


class CombineMessages:
    """An event-level model; all rows are opaque BF16 bytes until source reduce."""

    def __init__(self, ids, sequence, memory=None):
        self.ids = ids
        self.topk = ids.shape[2]
        self.sequence = sequence
        self.memory = memory if memory is not None else PersistentGatewayMemory()
        self.owners = {key: owner for key, owner, _ in routes(ids)}
        self.pending = {key for key, owner in self.owners.items() if host(key[0]) != host(owner)}
        self.inbox = {}
        self.outstanding = []
        self.puts = []
        self.produced = set()

    def produce(self, key, bf16_row):
        assert key not in self.produced
        self.produced.add(key)
        source, token, slot = key
        owner = self.owners[key]
        raw = np.asarray(bf16_row, dtype=np.float32).view(np.uint32)
        assert not np.any(raw & 0xFFFF)  # raw result is already BF16 rounded
        wire = (raw >> 16).astype(np.uint16).tobytes()
        if host(owner) == host(source):
            self.inbox[key] = wire
        else:
            destination = pair(source)
            assert host(destination) == host(owner)
            address = destination, token * self.topk + slot
            self.memory.rows[address] = wire
            self.memory.epochs[address] = self.sequence

    def poll(self):
        # Set intersection expresses readiness independently of the kernel's
        # per-token bit masks and sequential flag scanner.
        available = defaultdict(list)
        for source, token, slot in self.pending:
            row = token * self.topk + slot
            if self.memory.epochs.get((pair(source), row)) == self.sequence:
                available[source].append(row)
        posted = 0
        for source, rows in available.items():
            for _, group in groupby(enumerate(sorted(rows)), key=lambda item: item[1] - item[0]):
                run = tuple(row for _, row in group)
                keys = tuple((source, row // self.topk, row % self.topk) for row in run)
                payload = b"".join(self.memory.rows[pair(source), row] for row in run)
                self.outstanding.append((keys, payload))
                self.puts.append((pair(source), source, run))
                self.pending.difference_update(keys)
                posted += len(run)
        return posted

    def network_progress(self, rng):
        # Distinct runs may arrive in any order. A commit must follow completion
        # of all of them; poll/Put submission itself never fills the inbox.
        rng.shuffle(self.outstanding)
        for keys, payload in self.outstanding:
            width = len(payload) // len(keys)
            for index, key in enumerate(keys):
                assert key not in self.inbox
                self.inbox[key] = payload[index * width : (index + 1) * width]
        self.outstanding.clear()

    def source_results(self, weights, shared=None):
        assert not self.pending and not self.outstanding
        assert set(self.inbox) == set(self.owners)
        width = len(next(iter(self.inbox.values()))) // 2
        result = np.empty((*self.ids.shape[:2], width), np.float32)
        for source in range(WORLD):
            for token in range(self.ids.shape[1]):
                accumulator = None
                for slot in range(self.topk):
                    raw = np.frombuffer(self.inbox[source, token, slot], np.uint16).astype(np.uint32)
                    row = (raw << 16).view(np.float32)
                    weighted = np.multiply(row, weights[source, token, slot], dtype=np.float32)
                    accumulator = weighted if accumulator is None else np.add(accumulator, weighted, dtype=np.float32)
                if shared is not None:
                    accumulator = np.add(accumulator, shared[source, token], dtype=np.float32)
                result[source, token] = bf16(accumulator)
        return result


@pytest.mark.parametrize("bs", [1, 2, 7, 31])
@pytest.mark.parametrize("topk", [1, 8, 32])
@pytest.mark.parametrize("pattern", ["local", "remote", "mixed", "duplicate-rank"])
def test_hierarchical_messages_match_direct_routing(bs, topk, pattern):
    ids = routing(bs, topk, pattern)
    for token_ids in ids.reshape(-1, topk):
        assert len(set(token_ids)) == topk  # duplicate ranks still have distinct experts
    records = input_records(bs, seed=bs + topk)
    transfers, consumers = hierarchical_dispatch(ids, records)
    direct = {(owner, expert, key): records[key[:2]] for key, owner, expert in routes(ids)}
    assert consumers == direct
    expected_tokens = {
        (source, token, 1 - host(source))
        for source in range(WORLD)
        for token in range(bs)
        if any(host(int(expert) // EXPERTS_PER_RANK) != host(source) for expert in ids[source, token])
    }
    assert transfers == expected_tokens
    assert len(transfers) <= WORLD * bs
    if pattern == "duplicate-rank":
        assert len(transfers) == WORLD * bs
        assert all(len(set(token_ids // EXPERTS_PER_RANK)) == 1 for token_ids in ids.reshape(-1, topk))

    rng = np.random.default_rng(100 + bs + topk)
    contributions = bf16(rng.normal(size=(*ids.shape, 4)).astype(np.float32))
    weights = rng.uniform(-0.75, 1.25, size=ids.shape).astype(np.float32)
    shared = bf16(rng.normal(size=(*ids.shape[:2], 4)).astype(np.float32))
    oracle = np.multiply(contributions, weights[..., None], dtype=np.float32)
    expected = bf16(np.add(np.add.accumulate(oracle, axis=2, dtype=np.float32)[:, :, -1], shared, dtype=np.float32))

    model = CombineMessages(ids, sequence=137)
    arrivals = list(model.owners)
    scheduler = random.Random(bs * 1000 + topk)
    scheduler.shuffle(arrivals)
    for index, key in enumerate(arrivals):
        model.produce(key, contributions[key])
        if index % 37 == 0:
            model.poll()
            model.network_progress(scheduler)
    model.poll()
    model.network_progress(scheduler)
    assert not model.pending
    # Wire contents must equal the original GMM2 BF16 row, before weighting.
    for key, wire in model.inbox.items():
        raw = (contributions[key].view(np.uint32) >> 16).astype(np.uint16).tobytes()
        assert wire == raw
    counts = Counter((source, row // topk, row % topk) for _, source, run in model.puts for row in run)
    assert all(count == 1 for count in counts.values())
    assert set(counts) == {key for key, owner in model.owners.items() if host(key[0]) != host(owner)}
    assert all(destination == pair(gateway) for gateway, destination, _ in model.puts)
    put_count = len(model.puts)
    np.testing.assert_array_equal(model.source_results(weights, shared), expected)
    assert len(model.puts) == put_count  # shared expert accumulation is entirely local


def test_late_rows_do_not_block_ready_runs_or_merge_over_holes():
    ids = routing(1, 8, "remote")
    model = CombineMessages(ids, 4)
    value = bf16(np.array([0.25, -2.0], np.float32))
    for slot in (0, 1, 3, 4, 6, 7):
        model.produce((5, 0, slot), value)
    assert model.poll() == 6
    assert [run for _, source, run in model.puts if source == 5] == [(0, 1), (3, 4), (6, 7)]
    assert not model.inbox  # enqueued payload is not yet a completed receive
    assert model.poll() == 0
    model.network_progress(random.Random(4))
    assert set(model.inbox) == {(5, 0, slot) for slot in (0, 1, 3, 4, 6, 7)}
    for slot in (5, 2):
        model.produce((5, 0, slot), value)
        assert model.poll() == 1
    assert [run for _, source, run in model.puts if source == 5][-2:] == [(5,), (2,)]


def test_fp32_source_accumulation_keeps_topk_order():
    ids = routing(1, 8, "mixed")
    model = CombineMessages(ids, 9)
    rows = bf16(np.array([2**24, 1, -(2**24), 0, 0, 0, 0, 0], np.float32))
    for key in reversed(list(model.owners)):
        model.produce(key, rows[key[2] : key[2] + 1])
        model.poll()
    model.network_progress(random.Random(3))
    output = model.source_results(np.ones_like(ids, dtype=np.float32))
    np.testing.assert_array_equal(output, 0)
    regrouped = np.float32(np.float32(rows[0] + rows[2]) + rows[1])
    assert regrouped == 1  # arrival-order or host-local reduction changes the result


def test_repeated_calls_ignore_old_epoch_after_legacy_marker_wrap():
    memory = PersistentGatewayMemory()
    scheduler = random.Random(83)
    calls = [(31, 1, "remote"), (1, 128, "local"), (7, 129, "mixed"), (2, 256, "remote"), (31, 257, "duplicate-rank")]
    assert ((calls[0][1] & 127) | 128) == ((calls[-1][1] & 127) | 128)
    for bs, sequence, pattern in calls:
        ids = routing(bs, 8, pattern)
        model = CombineMessages(ids, sequence, memory)
        assert model.poll() == 0  # stale flags/rows from prior shapes cannot satisfy this call
        value = bf16(np.array([sequence, -sequence, bs], np.float32))
        arrivals = list(model.owners)
        scheduler.shuffle(arrivals)
        for key in arrivals:
            model.produce(key, value)
        model.poll()
        model.network_progress(scheduler)
        output = model.source_results(np.full(ids.shape, 0.125, np.float32))
        np.testing.assert_array_equal(output, np.broadcast_to(value, output.shape))


def lifecycle_dependencies(*, whole_host=True, wait_before_send=False):
    graph = {}
    for rank in range(WORLD):
        graph["produce", rank] = set()
    for machine in (0, 1):
        graph["host", machine] = {("produce", rank) for rank in range(WORLD) if host(rank) == machine}
    for rank in range(WORLD):
        finish = ("host", host(rank)) if whole_host else ("produce", rank)
        graph["post", rank] = {finish}
        graph["quiet", rank] = {("post", rank)}
        graph["commit", rank] = {("quiet", rank)}
        graph["commit-complete", rank] = {("commit", rank)}
        graph["receive", rank] = {("commit-complete", rank), ("commit", pair(rank))}
        graph["reuse", rank] = {("receive", rank), ("host", host(rank))}
        if wait_before_send:
            graph["post", rank].add(("receive", rank))
    return graph


def ancestors(graph, event):
    result = set()
    pending = list(graph[event])
    while pending:
        dependency = pending.pop()
        if dependency not in result:
            result.add(dependency)
            pending.extend(graph[dependency])
    return result


@pytest.mark.parametrize("late_rank", [0, 15, 16, 31])
@pytest.mark.parametrize("late_stage", ["produce", "commit"])
def test_host_finish_and_commit_close_buffer_lifetime_without_wait_cycle(late_rank, late_stage):
    graph = lifecycle_dependencies()
    finished = set()
    scheduler = random.Random(late_rank)
    while len(finished) < len(graph):
        runnable = [event for event, deps in graph.items() if event not in finished and deps <= finished]
        assert runnable, "outbound-before-inbound protocol must remain acyclic"
        ordinary = [event for event in runnable if event != (late_stage, late_rank)]
        event = scheduler.choice(ordinary or runnable)
        if event[0] == "reuse":
            assert {("produce", rank) for rank in range(WORLD)} <= finished
        if event[0] == "receive":
            assert ("commit-complete", event[1]) in finished
        finished.add(event)
    for rank in range(WORLD):
        assert {("produce", peer) for peer in range(WORLD)} <= ancestors(graph, ("reuse", rank))


def test_lifecycle_requires_even_zero_contribution_commit_and_full_host_finish():
    # A rank with no expected remote rows still closes the remote host's reads of
    # its old quantization/relay/rail buffers. Omitting its commit strands its peer.
    graph = lifecycle_dependencies()
    assert ("commit", 16) in ancestors(graph, ("reuse", 0))
    early = lifecycle_dependencies(whole_host=False)
    assert ("produce", 17) not in ancestors(early, ("reuse", 0))
    assert ("produce", 17) in ancestors(graph, ("reuse", 0))
    cyclic = lifecycle_dependencies(wait_before_send=True)
    assert ("receive", 0) in ancestors(cyclic, ("receive", 0))


def test_every_expert_source_pair_uses_only_local_mte_then_paired_rdma():
    for source in range(WORLD):
        assert pair(pair(source)) == source
        assert pair(source) % HOST_SIZE == source % HOST_SIZE
        for owner in range(WORLD):
            mte_destination = source if host(owner) == host(source) else pair(source)
            assert host(mte_destination) == host(owner)
            if mte_destination != source:
                assert pair(mte_destination) == source
