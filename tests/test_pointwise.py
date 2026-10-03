import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import pytest
import torch
from kernel.ops import extension
from quantdemoire.precision import configure
from helpers import synthetic_layer

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('shape',[(2,48,9,11),(1,64,1,1),(1,192,17,19)])
@torch.inference_mode()
def test_attention_rounding_and_graph(shape):
    configure();torch.manual_seed(2901)
    features=[torch.randn(shape,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last) for _ in range(4)]
    n,c,_,_=shape
    weights=torch.rand(n,3*c,1,1,device='cuda',dtype=torch.float16)
    def reference():
        a,b,c=weights.chunk(3,1)
        return features[3]+(features[0]*a+features[1]*b+features[2]*c)
    def fused():return extension().attention_residual(*features[:3],weights,features[3])
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):fused();fused()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=fused()
    for scale in (1e-5,1.,1000.):
        for x in features:x.normal_().mul_(scale)
        expected=reference();graph.replay()
        assert torch.equal(actual,expected)
        assert torch.equal(fused(),expected)


@pytest.mark.parametrize('bits',[4,8])
@pytest.mark.parametrize('stride,dilation',[(1,1),(2,3)])
@torch.inference_mode()
def test_sparse_residual(bits,stride,dilation):
    configure();torch.manual_seed(803)
    m=synthetic_layer(dict(weight=[36,48,3,3],stride=stride,padding=dilation,dilation=dilation),bits).cuda()
    m.main_half=True
    x=torch.randn(2,48,17,19,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    y=m(x);residual=torch.randn_like(y)
    assert torch.equal(m(x,residual=residual),y+residual)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=m(x,residual=residual)
    for scale in (.1,100.):
        residual.normal_().mul_(scale);x.normal_().mul_(scale)
        expected=m(x)+residual;graph.replay()
        assert torch.equal(actual,expected)


def test_attention_keeps_autograd_path():
    from quantdemoire.esdnet import ChannelScaleAttention
    configure();torch.manual_seed(809)
    module=ChannelScaleAttention(24).cuda().half().to(memory_format=torch.channels_last)
    module.fuse_attention=True
    features=[torch.randn(1,8,5,7,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last).requires_grad_() for _ in range(4)]
    result=module(*features[:3],residual=features[3])
    result.float().square().mean().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in features)
