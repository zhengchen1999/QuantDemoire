"""Linear-address CSR variants must preserve each pixel's FMA order."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import pytest
import torch
from helpers import synthetic_layer
from quantdemoire.precision import configure
from quantdemoire.deploy import ChannelConcat

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('bits',[4,8])
@pytest.mark.parametrize('geometry',[(2,17,19,1,1),(1,1,1,1,3),(2,17,1,2,2)])
@torch.inference_mode()
def test_tuned_csr(bits,geometry):
    configure();torch.manual_seed(872)
    n,h,w,stride,dil=geometry
    m=synthetic_layer(dict(weight=[36,48,3,3],stride=stride,padding=dil,dilation=dil),bits).cuda()
    m.main_half=True
    x=torch.randn(n,48,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    m.csr_kernel='legacy';expected=m(x);residual=torch.randn_like(expected)
    for mode in ('linear','lut','wide','linear32','divmod'):
        m.csr_kernel=mode
        assert torch.equal(m(x),expected)
        assert m.linear_offsets.numel()==m.values.numel()
        assert torch.equal(m(x,residual=residual),expected+residual)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):m(x);m(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=m(x,residual=residual)
        for scale in (.01,50.):
            x.normal_().mul_(scale);residual.normal_().mul_(scale)
            m.csr_kernel='legacy';reference=m(x)+residual
            m.csr_kernel=mode;graph.replay()
            assert torch.equal(captured,reference)
        m.csr_kernel='legacy';expected=m(x)
    # A different spatial shape must invalidate cached linear offsets.
    y=torch.randn(n,48,9,13,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    m.csr_kernel='legacy';reference=m(y)
    m.csr_kernel='wide';assert torch.equal(m(y),reference)


@pytest.mark.parametrize('bits,next_bits',[(4,4),(8,8),(4,8),(8,4)])
@pytest.mark.parametrize('shared_consumer',[False,True])
@torch.inference_mode()
def test_tuned_next_quant(bits,next_bits,shared_consumer):
    configure();torch.manual_seed(873)
    a=synthetic_layer(dict(weight=[32,48,3,3],stride=1,padding=2,dilation=2),bits).cuda()
    r=1 if shared_consumer else 3
    b=synthetic_layer(dict(weight=[32,80,r,r],stride=1,padding=r//2,dilation=1),next_bits).cuda()
    if shared_consumer:
        b.joint_warps=8;b.joint_variant=9
        b.shared_skip_planar=b.shared_skip_prefix_planar=True
    a.main_half=b.main_half=True
    x=torch.randn(2,48,13,19,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    def chain():
        y,pk,pl=a(x,next_quantizer=b)
        if shared_consumer and a.csr_kernel!='legacy':assert pl.untyped_storage().nbytes()==1
        return y,b(ChannelConcat([y,x],prepared=(b,pk,pl,32)))
    a.csr_kernel=b.csr_kernel='legacy';expected=chain()
    for mode in ('linear','lut','wide','linear32','divmod'):
        a.csr_kernel=b.csr_kernel=mode
        actual=chain()
        assert all(torch.equal(u,v) for u,v in zip(actual,expected))
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=chain()
        for scale in (.01,100.):
            x.normal_().mul_(scale)
            a.csr_kernel=b.csr_kernel='legacy';reference=chain()
            a.csr_kernel=b.csr_kernel=mode;graph.replay()
            assert all(torch.equal(u,v) for u,v in zip(captured,reference))
        a.csr_kernel=b.csr_kernel='legacy';expected=chain()
