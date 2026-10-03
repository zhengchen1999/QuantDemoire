import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import pytest
import torch
from kernel.ops import extension
from quantdemoire.precision import configure
from quantdemoire.deploy import pack_signed, unpack_signed, ChannelConcat
from helpers import synthetic_layer

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('bits',[4,8])
@torch.inference_mode()
def test_round_even_and_saturation(bits):
    configure()
    x=torch.tensor([-1000.,-.4375,-.3125,-.1875,-.0625,.0625,.1875,.3125,1000.],device='cuda',dtype=torch.float16)
    x=x.reshape(1,1,1,-1).expand(1,32,1,-1).contiguous(memory_format=torch.channels_last)
    a=torch.tensor([.125,3.],device='cuda');inv=torch.ones(32,device='cuda')
    packed,planar=extension().prepare(x,inv,a,bits,32)
    expected=((x.float()/.125+3).round().clamp(0,2**bits-1)-(1<<(bits-1))).to(torch.int8)
    assert torch.equal(planar,expected)
    actual=unpack_signed(packed,bits).reshape(1,1,9,32).permute(0,3,1,2)
    assert torch.equal(actual,expected)


@pytest.mark.parametrize('bits',[4,8])
@pytest.mark.parametrize('channels',[[12],[36,48],[32,32,32,48],[2,6],[3,5],[7]])
@torch.inference_mode()
def test_fused_concat_quantization(bits,channels):
    configure();torch.manual_seed(100)
    parts=[torch.randn(2,c,9,11,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last) for c in channels]
    c=sum(channels);align=32 if bits==4 else 16;cp=(c+align-1)//align*align
    inv=torch.rand(c,device='cuda')+.5
    a=torch.tensor([.07,3.],device='cuda')
    op=extension()
    expected=op.prepare(torch.cat(parts,1),inv,a,bits,cp)
    actual=op.prepare_cat(parts,inv,a,bits,cp)
    assert all(torch.equal(x,y) for x,y in zip(expected,actual))
    packed_only, empty_planar=op.prepare_cat(parts,inv,a,bits,cp,False)
    assert torch.equal(packed_only, actual[0])
    assert empty_planar.untyped_storage().nbytes() == 1
    q=(torch.cat(parts,1).float()*(inv/a[0])[None,:,None,None]+a[1]).round().clamp(0,2**bits-1)-(1<<(bits-1))
    assert torch.equal(q.to(torch.int8),actual[1])
    selected=torch.arange(1,c,3,device='cuda')
    mapping=torch.full((c,),-1,device='cuda',dtype=torch.int32)
    mapping[selected]=torch.arange(selected.numel(),device='cuda',dtype=torch.int32)
    compact=op.prepare_compact_cat(parts,inv,a,bits,cp,mapping,selected.numel())
    assert torch.equal(compact[0],actual[0])
    assert torch.equal(compact[1],actual[1][:,selected])
    assert compact[1].untyped_storage().nbytes()==2*selected.numel()*9*11
    for mode in (1,2,3,4,5):
        optimized=op.prepare_cached_cat(parts,inv/a[0],a,bits,cp,mapping,selected.numel(),mode,True)
        assert all(torch.equal(x,y) for x,y in zip(optimized,compact))
        packed_only,placeholder=op.prepare_cached_cat(parts,inv/a[0],a,bits,cp,mapping,selected.numel(),mode,False)
        assert torch.equal(packed_only,compact[0])
        assert placeholder.untyped_storage().nbytes()==1


@pytest.mark.parametrize('bits',[4,8])
@pytest.mark.parametrize('stride,dilation,zero',[(1,1,0),(1,3,3),(2,1,7)])
@torch.inference_mode()
def test_sparse_fusion_edges_and_tail_channels(bits,stride,dilation,zero):
    configure();torch.manual_seed(222)
    record=dict(weight=[36,48,3,3],stride=stride,padding=dilation,dilation=dilation)
    m=synthetic_layer(record,bits,zero_point=zero).cuda()
    m.compact_sparse=False
    x=torch.randn(2,48,17,19,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    m.fused=True;fused=m(x)
    m.fused=False;separate=m(x)
    reference=m.fake_forward(x)
    torch.testing.assert_close(fused.float(),reference,rtol=.002,atol=.002)
    torch.testing.assert_close(fused,separate,rtol=.002,atol=.002)
    m.make_sparse_blocks();m.sparse_impl='tensorcore'
    tensorcore=m(x)
    torch.testing.assert_close(tensorcore.float(),reference,rtol=.002,atol=.002)
    m.sparse_impl='csr'
    # Duplicate buffers on a non-default stream to catch wrong-stream launches.
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual=m(x)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual,separate,rtol=0,atol=0)
    m.main_half=True
    half_main=m(x)
    half_reference=m.fake_forward(x)
    torch.testing.assert_close(half_main.float(),half_reference,rtol=.002,atol=.002)
    m.make_compact_sparse();m.compact_sparse=True
    assert torch.equal(m(x),half_main)
    m.main_half=False
    assert torch.equal(m(x),separate)
    m.fused=True
    assert torch.equal(m(x),fused)
    for implementation in ('cached','pair','quad','quad_loop','quad_swizzle'):
        m.prepare_impl=implementation
        assert torch.equal(m(x),fused)
        m.fused=False;m.main_half=True
        assert torch.equal(m(x),half_main)
        m.fused=True;m.main_half=False
