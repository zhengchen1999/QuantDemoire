import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import pytest
import torch
from helpers import synthetic_layer
from quantdemoire.precision import configure

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('bits',[4,8])
@pytest.mark.parametrize('warps,variant',[(4,0),(8,0),(8,1),(8,2),(8,3),(8,4),(8,5),(8,6),(8,7),(8,8),(8,9)])
@pytest.mark.parametrize('shape',[(36,48,3,2,3,17,19),(32,80,3,1,1,17,19),(64,144,1,1,1,9,11),(12,64,3,1,1,1,1),(96,80,3,1,2,13,21),(96,80,1,1,1,9,11)])
@torch.inference_mode()
def test_joint_output(bits,warps,variant,shape):
    configure();torch.manual_seed(819)
    k,c,r,stride,dil,h,w=shape
    m=synthetic_layer(dict(weight=[k,c,r,r],stride=stride,padding=dil if r==3 else 0,dilation=dil),bits).cuda()
    m.main_half=True;m.csr_kernel='linear';m.spec['relu']=True
    m.joint_variant=variant
    x=torch.randn(2,c,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    for sparse in (True,False):
        m.sparse_enabled=sparse;m.joint_warps=0;expected=m(x)
        m.joint_warps=warps;actual=m(x)
        assert torch.equal(actual,expected),float((actual-expected).abs().max())

    m.sparse_enabled=True;m.joint_warps=0;expected=m(x);residual=torch.randn_like(expected)
    m.joint_warps=warps
    assert torch.equal(m(x,residual=residual),expected+residual)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):m(x);m(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=m(x,residual=residual)
    for scale in (.1,50.):
        x.normal_().mul_(scale);residual.normal_().mul_(scale)
        m.joint_warps=0;expected=m(x)+residual
        m.joint_warps=warps;graph.replay()
        assert torch.equal(actual,expected),float((actual-expected).abs().max())


@pytest.mark.parametrize('bits,c',[(4,80),(8,80),(4,240),(8,144)])
@torch.inference_mode()
def test_shared_pointwise_without_planar(bits,c):
    configure();torch.manual_seed(534)
    m=synthetic_layer(dict(weight=[96,c,1,1],stride=1,padding=0,dilation=1),bits).cuda()
    m.main_half=True;m.csr_kernel='linear';m.prepare_impl='quad_swizzle'
    m.joint_warps=8;m.joint_variant=9
    x=torch.randn(2,c,9,11,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    m.shared_skip_planar=False;expected=m(x)
    m.shared_skip_planar=True;actual=m(x)
    assert torch.equal(actual,expected)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured=m(x)
    for scale in (.1,40.):
        x.normal_().mul_(scale)
        m.shared_skip_planar=False;expected=m(x)
        m.shared_skip_planar=True;graph.replay()
        assert torch.equal(captured,expected)
