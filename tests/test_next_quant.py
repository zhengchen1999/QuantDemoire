"""Producer epilogues must preserve half rounding and consumer-specific state."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import pytest
import torch
from kernel.ops import extension
from quantdemoire.deploy import ChannelConcat
from quantdemoire.precision import configure
from helpers import synthetic_layer

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('bits,next_bits',[(4,4),(8,8),(4,8),(8,4)])
@pytest.mark.parametrize('n,h,w,relu',[(2,9,11,True),(1,1,1,False)])
@pytest.mark.parametrize('joint_consumer,joint_variant',[(0,0),(4,0),(8,0),(8,1),(8,2),(8,3),(8,4),(8,5),(8,6),(8,7),(8,8),(8,9)])
@pytest.mark.parametrize('prepare_impl',['pair','quad','quad_loop','quad_swizzle'])
@torch.inference_mode()
def test_next_quant_chain(bits,next_bits,n,h,w,relu,joint_consumer,joint_variant,prepare_impl):
    configure();torch.manual_seed(741)
    layers=[]
    for c,r,d,b in [(48,3,3,bits),(80,3,1,next_bits),(112,1,1,bits)]:
        m=synthetic_layer(dict(weight=[32,c,r,r],stride=1,padding=d if r==3 else 0,dilation=d),b)
        m.main_half=True;m.spec['relu']=relu;m.prepare_impl=prepare_impl
        layers.append(m.cuda())
    a,b,c=layers
    c.joint_warps=joint_consumer
    c.joint_variant=joint_variant
    c.shared_skip_planar=joint_variant==9
    c.shared_skip_prefix_planar=joint_variant==9
    if joint_variant==9:b.csr_kernel="linear"
    x=torch.randn(n,48,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    op=extension()

    def reference():
        y=a(x);z=b(ChannelConcat([y,x]))
        return y,z,c(ChannelConcat([z,y,x]))

    def fused(check=False):
        y,pk,pl=a(x,next_quantizer=b)
        parts=[y,x]
        if check:
            expected=op.prepare_cached_cat(parts,b.quant_multiplier,b.activation,next_bits,b.spec['padded_channels'],b.sparse_channel_map,b.sparse_channels,2,True)
            actual=op.prepare_next_cat(parts,b.quant_multiplier,b.activation,next_bits,b.spec['padded_channels'],b.sparse_channel_map,b.sparse_channels,pk,pl,32)
            assert all(torch.equal(u,v) for u,v in zip(expected,actual))
        first=ChannelConcat(parts,prepared=(b,pk,pl,32))
        z,pk2,pl2=b(first,next_quantizer=c)
        if c.shared_skip_prefix_planar and c.uses_shared_pointwise():
            assert pl2.untyped_storage().nbytes()==1
        last=ChannelConcat([z,first],prepared=(c,pk2,pl2,32))
        # An old concat's preparation must not leak into another consumer.
        assert ChannelConcat([z,first]).prepared is None
        return y,z,c(last)

    expected=reference();actual=fused(check=True)
    assert all(torch.equal(u,v) for u,v in zip(expected,actual))
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused();fused()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured=fused()
    for scale in (.1,10.,1000.):
        x.normal_().mul_(scale)
        expected=reference();graph.replay()
        assert all(torch.equal(u,v) for u,v in zip(expected,captured))

    # Wrong consumer reuse is an error, even if shapes happen to agree.
    y,pk,pl=a(x,next_quantizer=b)
    with pytest.raises(ValueError,match='different consumer'):
        b(ChannelConcat([y,x],prepared=(a,pk,pl,32)))
