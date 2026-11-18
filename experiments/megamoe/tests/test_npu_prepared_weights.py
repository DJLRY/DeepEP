"""Single-device EPLB checks against real NZ storage and a captured table reader."""

import pytest


def test_npu_eplb_layout_refresh_and_graph_table_replacement():
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an available Ascend NPU")
    from deep_ep_experimental.megamoe import prepare_weights_for_mega_moe

    def raw(n, k, seed):
        gen = torch.Generator().manual_seed(seed)

        def data(shape):
            return torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen).npu()

        return data((n, k // 2)), data((n, k // 64, 2))

    first = [raw(512, 1024, i) for i in (1, 2)]
    second = [raw(1024, 256, i) for i in (3, 4)]
    owner = prepare_weights_for_mega_moe(first, second)
    planar, _ = owner.expert_views(layout="planar")
    fused, _ = owner.expert_views(layout="fused")
    addresses = [t.data_ptr() for pair in (planar, fused) for group in pair for t in group]

    def read_weight(t):
        assert torch_npu.get_npu_format(t) in (29, 50)
        return torch_npu.npu_format_cast(
            t, 2, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
        ).cpu()

    # Modify both the packed FP4 bytes and E8M0 bytes through exported zero-copy views.
    for group in planar:
        group[1].copy_(group[0])
    owner.refresh_gmm1([1], source_layout="planar")
    for i, t in enumerate((read_weight(fused[0][1]), fused[1][1].view(torch.uint8).cpu())):
        gate0, gate1, up0, up1 = first[0][i].cpu().split(128, dim=0)
        assert torch.equal(t, torch.cat((gate0, up0, gate1, up1)))
    owner.refresh_gmm1([1], source_layout="fused")
    assert torch.equal(read_weight(planar[0][1]), first[0][0].cpu())
    assert torch.equal(planar[1][1].view(torch.uint8).cpu(), first[0][1].cpu())
    assert addresses == [t.data_ptr() for pair in (planar, fused) for group in pair for t in group]

    tables = [owner.select(layout)[2] for layout in (True, False)]
    before = [t.cpu() for t in tables]
    saved = [torch.empty_like(t) for t in tables]
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for dst, src in zip(saved, tables):
            dst.copy_(src)
    owner.update_experts([1], [raw(512, 1024, 5)], [raw(1024, 256, 6)], replace=True)
    graph.replay()
    torch.npu.synchronize()
    for idx, (src, dst, old) in enumerate(zip(tables, saved, before)):
        assert owner.select(idx == 0)[2].data_ptr() == src.data_ptr()
        assert torch.equal(dst.cpu(), src.cpu())
        assert torch.equal(dst.cpu()[:, 0], old[:, 0])
        assert torch.all(dst.cpu()[:, 1] != old[:, 1])
    owner.release_retired()
