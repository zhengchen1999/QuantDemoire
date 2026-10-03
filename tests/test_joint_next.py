"""Joint producers preserve output-half rounding and consumer-specific prefixes."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import pytest
import torch
from kernel.ops import extension
from helpers import synthetic_layer
from quantdemoire.deploy import ChannelConcat
from quantdemoire.precision import configure

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

@pytest.mark.parametrize('bits,next_bits',[(4,4),(4,8),(8,4),(8,8)])
@pytest.mark.parametrize('r',[1,3])
@pytest.mark.parametrize('shared',[False,True])
@pytest.mark.parametrize('geometry',[(2,17,19,2),(1,1,1,1)])
@pytest.mark.parametrize('strategy',[0,1])
@pytest.mark.parametrize('producer_prefixed',[False,True])
@torch.inference_mode()
def test_joint_next(bits,next_bits,r,shared,geometry,strategy,producer_prefixed):
    configure();op=extension()
    torch.manual_seed(855)
    n,h,w,dil=geometry;pad=dil if r==3 else 0
    a=synthetic_layer(dict(weight=[32,48,r,r],stride=1,padding=pad,dilation=dil),bits).cuda()
    cr=1 if shared else 3
    b=synthetic_layer(dict(weight=[32,80,cr,cr],stride=1,padding=cr//2,dilation=1),next_bits).cuda()
    for m in (a,b):m.main_half=True;m.csr_kernel='divmod';m.prepare_impl='quad_swizzle';m.spec['relu']=True;m.joint_warps=8
    a.joint_variant=9 if r==1 else 8
    a.joint_next_mode=strategy+1
    a.shared_skip_planar=r==1
    b.joint_variant=9 if shared else 8;b.shared_skip_planar=shared;b.shared_skip_prefix_planar=shared
    x=torch.randn(n,48,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    def reference():
        y=a(x)
        return y,b(ChannelConcat([y,x]))
    def fused(check=False):
        source=x
        if producer_prefixed:
            parts=[x[:,:16].contiguous(memory_format=torch.channels_last),x[:,16:].contiguous(memory_format=torch.channels_last)]
            pk,pl=op.prepare_cached_cat(parts,a.quant_multiplier,a.activation,bits,a.spec['padded_channels'],a.sparse_channel_map,a.sparse_channels,5,r!=1)
            source=ChannelConcat(parts,prepared=(a,pk,pl,16))
        y,packed,planar=a(source,next_quantizer=b)
        if check:
            full=op.prepare_cached_cat([y,x],b.quant_multiplier,b.activation,next_bits,b.spec['padded_channels'],b.sparse_channel_map,b.sparse_channels,5,True)
            actual=op.prepare_next_cat([y,x],b.quant_multiplier,b.activation,next_bits,b.spec['padded_channels'],b.sparse_channel_map,b.sparse_channels,packed,planar,32,5,not shared)
            assert torch.equal(actual[0],full[0])
            if shared:assert planar.untyped_storage().nbytes()==1
            else:assert torch.equal(actual[1],full[1])
        return y,b(ChannelConcat([y,x],prepared=(b,packed,planar,32)))
    expected=reference();actual=fused(True)
    assert all(torch.equal(u,v) for u,v in zip(expected,actual)),(bits,next_bits,r,shared,geometry)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):fused();fused()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured=fused()
    for scale in (.1,100.):
        x.normal_().mul_(scale);expected=reference();graph.replay()
        assert all(torch.equal(u,v) for u,v in zip(expected,captured)),(bits,next_bits,r,shared,geometry,'graph')
