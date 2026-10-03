import pytest
import torch
from helpers import synthetic_layer
from research.kernel import planar_int4
from kernel.ops import extension
from quantdemoire.deploy import balance_csr_rows,unpack_signed
from quantdemoire.precision import configure

pytestmark=[pytest.mark.experimental,pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')]


@pytest.mark.parametrize('warps',[0,8])
@pytest.mark.parametrize('geometry',[(1,19,18,1,1),(2,17,22,3,1),(1,9,8,2,2)])
@pytest.mark.parametrize('segments',[1,3])
@pytest.mark.parametrize('residual',[False,True])
@torch.inference_mode()
def test_planar_int4(warps,geometry,segments,residual):
    configure();torch.manual_seed(131)
    n,h,w,dilation,stride=geometry
    m=synthetic_layer(dict(weight=[32,96,3,3],stride=stride,padding=dilation,dilation=dilation),4).cuda()
    m.main_half=True;m.joint_warps=warps;m.joint_variant=9
    m.balance_csr=True;m.balanced_row=balance_csr_rows(m.row);m.packed_csr_meta=True
    m.prepare_impl='quad_swizzle';m.csr_kernel='divmod';m.spec['relu']=True
    x=torch.randn(n,96,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    parts=[part.contiguous(memory_format=torch.channels_last) for part in x.chunk(segments,1)]
    expected_packed,expected_planar=extension().prepare_cached_cat(parts,m.quant_multiplier,m.activation,4,
        m.spec['padded_channels'],m.sparse_channel_map,m.sparse_channels,5,True)
    packed,planar=planar_int4.prepare(m,parts)
    assert torch.equal(packed,expected_packed)
    assert planar.untyped_storage().nbytes()*2==expected_planar.untyped_storage().nbytes()
    raw=planar.as_strided((planar.numel()//2,),(1,)).view(torch.uint8)
    assert torch.equal(unpack_signed(raw,4).reshape_as(expected_planar),expected_planar)
    addition=torch.randn_like(m(x)) if residual else None
    def check(actual):
        expected=m(torch.cat(parts,1).contiguous(memory_format=torch.channels_last))
        if addition is not None:expected=expected+addition
        assert torch.equal(actual,expected),(actual-expected).abs().max()
    def run():
        packed,planar=planar_int4.prepare(m,parts)
        return planar_int4.convolution(m,packed,planar,residual=addition)
    check(run())
    run();cached_offsets=m.linear_offsets;run()
    assert m.linear_offsets is cached_offsets
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):run();run()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=run()
    for part in parts:part.normal_().mul_(5)
    graph.replay();check(out)
