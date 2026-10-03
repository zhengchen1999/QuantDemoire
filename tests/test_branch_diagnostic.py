"""The timing counterfactual must preserve dense values through all caches."""
import pytest
import torch
from helpers import synthetic_layer
from quantdemoire.branch_diagnostic import main_only
from quantdemoire.deploy import ChannelConcat, balance_csr_rows
from quantdemoire.precision import configure

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('bits', [4, 8])
@pytest.mark.parametrize('warps,channels,r,omit,second', [
    (0, 96, 3, False, False), (0, 96, 3, True, False),
    (8, 32, 3, False, True), (8, 96, 3, False, True),
    (8, 96, 3, True, True), (8, 96, 1, False, False),
])
@torch.inference_mode()
def test_main_only_caches(bits, warps, channels, r, omit, second):
    configure(); torch.manual_seed(742)
    layers = torch.nn.ModuleList([
        synthetic_layer(dict(weight=[32, channels + 32*i, r, r],
                             stride=1, padding=r//2, dilation=1), bits).cuda()
        for i in range(3)])
    for m in layers:
        m.main_half = True; m.csr_kernel = 'divmod'; m.prepare_impl = 'quad_swizzle'
        m.spec['relu'] = True; m.joint_warps = warps; m.joint_variant = 9
        m.joint_next_mode = 2; m.balance_csr = True; m.packed_csr_meta = True
        m.balanced_row = balance_csr_rows(m.row)
        m.few_channels = True; m.fuse_second_quant = second
        m.shared_skip_planar = True; m.shared_skip_prefix_planar = True
        m.retain_shared = True
    a, b, c = layers
    x = torch.randn(2, channels, 17, 19, device='cuda', dtype=torch.float16).contiguous(memory_format=torch.channels_last)

    def reference():
        saved = [(m.sparse_enabled, m.diagnostic_main_only) for m in layers]
        try:
            for m in layers:
                m.sparse_enabled = False; m.diagnostic_main_only = False
            y = a(x); z = b(ChannelConcat([y, x]))
            return c(ChannelConcat([z, y, x])) if second else z
        finally:
            for m, (sparse, diagnostic) in zip(layers, saved):
                m.sparse_enabled = sparse; m.diagnostic_main_only = diagnostic

    def fused():
        first = a(x, next_quantizer=b, future_quantizer=c if second else None, omit_feature=omit)
        y, pk, pl = first[:3]
        assert pl.untyped_storage().nbytes() == 1
        if not second:
            return b(ChannelConcat([y, x], prepared=(b, pk, pl, 32)))
        z, pk2, pl2 = b(ChannelConcat([y, x], prepared=(b, pk, pl, 32)), next_quantizer=c)
        return c(ChannelConcat([z, y, x], prepared=(c, pk2, pl2, 32),
                               second_cached=first[3] if second else None))

    expected = reference()
    with main_only(layers):
        actual = fused()
        assert torch.equal(expected, actual), (expected-actual).abs().max()
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fused(); fused()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fused()
        x.normal_(); expected = reference(); graph.replay()
        assert torch.equal(expected, captured), (expected-captured).abs().max()
    assert all(m.sparse_enabled and not m.diagnostic_main_only for m in layers)


@pytest.mark.parametrize('bits,warps', [(4, 0), (4, 8), (8, 0), (8, 8)])
@torch.inference_mode()
def test_main_only_residual(bits, warps):
    configure(); torch.manual_seed(99)
    m = synthetic_layer(dict(weight=[32, 48, 3, 3], stride=1, padding=1, dilation=1), bits).cuda()
    m.main_half = True; m.prepare_impl = 'quad_swizzle'; m.joint_warps = warps
    m.joint_variant = 9; m.spec['relu'] = True
    x = torch.randn(1, 48, 19, 17, device='cuda', dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    residual = torch.randn(1, 32, 19, 17, device='cuda', dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    m.sparse_enabled = False
    expected = m(x) + residual
    with main_only(m):
        assert torch.equal(m(x, residual=residual), expected)
    assert not m.sparse_enabled and not m.diagnostic_main_only
