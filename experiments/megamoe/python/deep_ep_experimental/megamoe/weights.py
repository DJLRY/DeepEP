"""Prepared expert weights. Updates run outside capture under caller-owned stream ordering."""

from .api import _SharedFp8Pair, _check_fp4_weight_format, _interleave_gate_up_128, transform_weights_for_mega_moe


class WeightHandle:
    def __init__(self, owner):
        self.owner = owner


def _reorder(pair, inverse):
    import torch
    import torch_npu

    weight, scale = pair
    nd = torch_npu.npu_format_cast(
        weight, 2, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
    )

    def reorder(t):
        if not inverse:
            return _interleave_gate_up_128(t)
        e, n, *tail = t.shape
        return t.reshape((e, n // 256, 2, 128, *tail)).transpose(1, 2).contiguous().reshape(t.shape)

    nz = torch_npu.npu_format_cast(
        reorder(nd), 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
    )
    return nz, reorder(scale.view(torch.uint8)).view(torch.float8_e8m0fnu)


def _validate(l1, l2, planar):
    import torch
    import torch_npu

    w1, w2 = l1[0], l2[0]
    fp4 = w1.dtype == torch.uint8
    if w1.ndim != 3 or w2.ndim != 3 or w2.dtype != w1.dtype or (not fp4 and w1.dtype != torch.float8_e4m3fn):
        raise ValueError("invalid prepared projection shape or dtype")
    e, n, packed = w1.shape
    h = packed * (2 if fp4 else 1)
    if min(e, n, h) <= 0 or n % 256 or h % 64 or tuple(w2.shape) != (e, h, n // (4 if fp4 else 2)):
        raise ValueError("incompatible prepared projection dimensions")
    for w, scale in (l1, l2, planar):
        if fp4:
            _check_fp4_weight_format(w, "prepared")
        elif torch_npu.get_npu_format(w) not in (0, 2):
            raise ValueError("FP8 weights require ND storage")
        if (
            w.device.type != "npu"
            or w.device != w1.device
            or scale.device != w.device
            or not w.is_contiguous()
            or not scale.is_contiguous()
            or scale.dtype != torch.float8_e8m0fnu
            or tuple(scale.shape) != (*w.shape[:2], (w.shape[-1] * (2 if fp4 else 1) + 63) // 64, 2)
        ):
            raise ValueError("invalid prepared weight or E8M0 scale layout")
    if planar[0].shape != w1.shape or planar[0].dtype != w1.dtype:
        raise ValueError("planar and fused GMM1 must match")


def _prepare(l1, l2, shared, layout, planar_l1):
    import torch

    def pair(value):
        w, s = value
        return w, s.view(torch.float8_e8m0fnu) if s.dtype == torch.uint8 else s

    l1, l2 = pair(l1), pair(l2)
    dual = not shared and l1[0].dtype in (torch.uint8, torch.int8)
    if layout == "nd":
        if planar_l1 is not None:
            raise ValueError("planar_l1 belongs to prepared input")
        l1, l2 = transform_weights_for_mega_moe(l1, l2, shared=shared)
        planar = getattr(l1, "_planar", l1)
    elif layout == "planar_nz":
        if l1[0].dtype != torch.uint8 or planar_l1 is not None:
            raise ValueError("planar_nz requires packed NZ weights")
        planar = l1
        _validate(planar, l2, planar)
        if dual:
            l1 = _reorder(planar, False)
    elif layout == "prepared":
        if dual != (planar_l1 is not None):
            raise ValueError("prepared routed FP4 requires planar_l1")
        planar = pair(planar_l1) if dual else l1
    else:
        raise ValueError("input_layout must be nd, planar_nz or prepared")
    _validate(l1, l2, planar)
    l1 = _SharedFp8Pair(l1) if shared and l1[0].dtype == torch.float8_e4m3fn else tuple(l1)
    return l1, tuple(l2), tuple(planar)


def _expert_group(pair):
    """A list of (weight, scale) or a tuple of parallel tensor lists."""
    if isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], list):
        return list(zip(*pair, strict=True))
    return pair if isinstance(pair, list) else None


def _single(pair, layout):
    w, s = pair
    if w.ndim != 2 or s.ndim not in (2, 3):
        raise ValueError("list entries require a single 2-D expert weight")
    import torch

    if layout != "nd" and w.dtype == torch.uint8 and w.storage_offset() != 0:
        raise ValueError("prepared list weights require independent zero-offset storage")
    return w.unsqueeze(0), s.unsqueeze(0)


def _records(l1, l2, shared, layout, planar):
    groups = _expert_group(l1), _expert_group(l2)
    if groups == (None, None):
        return False, [_prepare(l1, l2, shared, layout, planar)]
    if any(g is None or not len(g) for g in groups):
        raise ValueError("supply complete nonempty lists for both projections")
    planes = [None] * len(groups[0]) if planar is None else _expert_group(planar)
    if planes is None:
        raise ValueError("planar_l1 must match the routed list")
    records = [
        _prepare(_single(x, layout), _single(y, layout), shared, layout, _single(p, layout) if p is not None else None)
        for x, y, p in zip(*groups, planes, strict=True)
    ]
    signature = _signature(records[0])
    if any(_signature(record) != signature for record in records):
        raise ValueError("all list experts must share one shape, dtype, stride and device")
    _check_nz_storage(records)
    return True, records


def _check_nz_storage(records):
    import torch

    if records[0][0][0].dtype == torch.uint8:
        for row in range(3):
            weights = [record[row][0] for record in records]
            if any(w.storage_offset() for w in weights) or len(
                {w.untyped_storage().data_ptr() for w in weights}
            ) != len(weights):
                raise ValueError("each NZ expert weight needs independent zero-offset storage")


def _signature(record):
    return tuple((t.shape, t.dtype, t.device, t.stride()) for pair in record for t in pair)


class PreparedWeights:
    def __init__(self, l1, l2, shared, layout, planar):
        self.shared = shared
        self.list_mode, self._records = _records(l1, l2, shared, layout, planar)
        self.count = len(self._records) if self.list_mode else self._records[0][0][0].shape[0]
        self.l1, self.l2 = WeightHandle(self), WeightHandle(self)
        self._tables = {}
        self._retired = []
        self._valid = True
        self._pending = set()
        self._dirty_tables = False
        if self.list_mode:
            self._publish_tables()
        self.commit_prepared_update()

    def _publish_tables(self):
        import torch

        for fused, index in ((True, 0), (False, 2)):
            rows = [[r[p][s].data_ptr() for r in self._records] for p, s in ((index, 0), (1, 0), (index, 1), (1, 1))]
            host = torch.tensor(rows, dtype=torch.int64, device="cpu", pin_memory=True)
            self._retired.append(host)
            if fused not in self._tables:
                self._tables[fused] = torch.empty_like(host, device=self._records[0][0][0].device)
            self._tables[fused].copy_(host, non_blocking=True)

    def select(self, fused):
        if not self._valid:
            raise RuntimeError("weight update incomplete")
        first, second, planar = self._records[0]
        return (first if fused or self.shared else planar), second, self._tables.get(fused)

    def expert_views(self, expert):
        if not 0 <= expert < self.count:
            raise IndexError("expert out of range")
        record = self._records[expert if self.list_mode else 0]
        i = 0 if self.list_mode else expert
        views = {}
        for name, pair in zip(("l1", "l2", "planar_l1"), record):
            slices = tuple(t[i] for t in pair)
            if any(v.untyped_storage().data_ptr() != t.untyped_storage().data_ptr() for v, t in zip(slices, pair)):
                raise RuntimeError("backend did not return a zero-copy expert view")
            views[name] = slices
        return views

    def _versions(self):
        return tuple(t._version if not t.is_inference() else None for r in self._records for p in r for t in p)

    def check_versions(self):
        if not self._valid or self._versions() != self._snapshot:
            raise RuntimeError("expert weights changed without a complete update")
        return True

    def commit_prepared_update(self):
        """Publish a complete externally ordered update to all weight representations."""
        if self._dirty_tables:
            self._publish_tables()
            self._dirty_tables = False
        self._snapshot = self._versions()
        self._pending.clear()
        self._valid = True

    def release_retired(self):
        """Call only after the framework has drained older readers and copies."""
        self._retired.clear()

    def _update(self, indices, records, replace):
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(self._records) for i in indices)
        ):
            raise ValueError("invalid updated expert indices")
        for i, r in zip(indices, records, strict=True):
            if _signature(self._records[i]) != _signature(r):
                raise ValueError("replacement must preserve weight layout")
        if replace:
            future = self._records.copy()
            for i, r in zip(indices, records):
                future[i] = r
            _check_nz_storage(future)
        if self._pending.difference(indices):
            raise RuntimeError("retry all experts from the incomplete update")
        self._valid = False
        self._pending.update(indices)
        self._dirty_tables |= replace
        self._retired.extend(records)
        for i, new in zip(indices, records):
            old = self._records[i]
            if replace:
                self._retired.append(old)
                self._records[i] = new
            else:
                for dst, src in zip(old, new):
                    for d, s in zip(dst, src):
                        d.copy_(s)
        self.commit_prepared_update()

    def update(self, l1, l2, *, input_layout="nd", planar_l1=None):
        if self.list_mode:
            raise ValueError("use update_experts on a list owner")
        self._update([0], [_prepare(l1, l2, self.shared, input_layout, planar_l1)], False)

    def update_experts(self, experts, l1, l2, *, replace=False, input_layout="nd", planar_l1=None):
        if not self.list_mode:
            raise ValueError("update_experts requires a list owner")
        is_list, records = _records(l1, l2, self.shared, input_layout, planar_l1)
        if not is_list:
            raise ValueError("supply lists of changed expert tensors")
        self._update(list(experts), records, replace)

    def refresh_gmm1(self, source="planar", experts=None):
        if source not in ("planar", "fused"):
            raise ValueError("source must be planar or fused")
        indices = list(range(self.count) if experts is None else experts)
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < self.count for i in indices)
        ):
            raise ValueError("invalid expert indices")
        if not self._valid:
            raise RuntimeError("restore the incomplete weight update before refreshing GMM1")
        self._valid = False
        self._pending.update(indices if self.list_mode else [0])
        for i in indices:
            views = self.expert_views(i)
            first, planar = views["l1"], views["planar_l1"]
            if first[0].data_ptr() == planar[0].data_ptr():
                continue
            src, dst = (planar, first) if source == "planar" else (first, planar)
            updated = _reorder(tuple(t.unsqueeze(0) for t in src), source == "fused")
            self._retired.append(updated)
            for d, s in zip(dst, updated):
                d.copy_(s[0])
        self.commit_prepared_update()


def prepare_weights_for_mega_moe(l1_weights, l2_weights, *, shared=False, input_layout="nd", planar_l1=None):
    return PreparedWeights(l1_weights, l2_weights, shared, input_layout, planar_l1)
