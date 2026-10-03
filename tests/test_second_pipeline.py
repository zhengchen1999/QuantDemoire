"""DenseBlock owns and consumes each second-successor cache once."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import pytest
import torch
from helpers import synthetic_layer
from quantdemoire.esdnet import DenseBlock
from quantdemoire.deploy import ChannelConcat,PackedConv
from quantdemoire.precision import configure

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

@pytest.mark.parametrize('bits',[4,8])
@pytest.mark.parametrize('geometry',[(2,17,19),(1,1,1)])
@pytest.mark.parametrize('middle_joint',[False,True])
@pytest.mark.parametrize('post_shared',[False,True])
@torch.inference_mode()
def test_dense_chain(bits,geometry,middle_joint,post_shared,monkeypatch):
    for key,value in dict(QUANTDEMOIRE_MAIN_HALF='1',QUANTDEMOIRE_BALANCE_CSR='1',
        QUANTDEMOIRE_PACKED_CSR_META='1',QUANTDEMOIRE_FUSE_SECOND_QUANT='1',
        QUANTDEMOIRE_PREPARE_IMPL='quad_swizzle',QUANTDEMOIRE_JOINT_WARPS='8',
        QUANTDEMOIRE_JOINT_VARIANT='9',QUANTDEMOIRE_JOINT_NEXT_QUANT='2',
        QUANTDEMOIRE_CSR_KERNEL='divmod',QUANTDEMOIRE_FEW_CHANNELS='1').items():
        monkeypatch.setenv(key,value)
    configure();torch.manual_seed(522)
    block=DenseBlock(48,(1,2,1),32).cuda().half().eval()
    block.lazy_concat=True;block.fuse_next_quant=True
    sequence=[]
    for i,layer in enumerate(block.conv_layers):
        dilation=(1,2,1)[i]
        m=synthetic_layer(dict(weight=[32,48+32*i,3,3],stride=1,padding=dilation,dilation=dilation),bits).cuda()
        m.spec['relu']=True;layer.conv[0]=m;layer.conv[1]=torch.nn.Identity();sequence.append(m)
    post=synthetic_layer(dict(weight=[48,144,1,1],stride=1,padding=0,dilation=1),bits).cuda()
    post.shared_skip_planar=post_shared;post.shared_skip_prefix_planar=post_shared
    if not post_shared:post.joint_variant=8
    post.retain_shared=True
    block.conv_post.conv=post;sequence.append(post)
    if not middle_joint:sequence[1].joint_warps=0
    n,h,w=geometry
    x=torch.randn(n,48,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    residual=torch.randn_like(x)
    def reference():
        for m in sequence:m.fuse_second_quant=False
        y=block(x,residual=residual)
        for m in sequence:m.fuse_second_quant=True
        return y
    observed=[]
    def prehook(m,args):
        value=args[0]
        if isinstance(value,ChannelConcat) and value.second_cached is not None:
            assert value.prepared is not None and value.prepared[0] is m
            observed.append(value)
    handles=[m.register_forward_pre_hook(prehook) for m in sequence]
    expected=reference();actual=block(x,residual=residual)
    assert torch.equal(actual,expected)
    assert len(observed)==(2 if middle_joint else 1)
    assert all(value.prepared is None and value.second_cached is None for value in observed)
    for handle in handles:handle.remove()
    observed.clear()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):block(x,residual=residual);block(x,residual=residual)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured=block(x,residual=residual)
    for scale in (.01,100.):
        x.normal_().mul_(scale);expected=reference();graph.replay()
        assert torch.equal(captured,expected)


def test_concat_does_not_inherit_second_cache():
    x=torch.zeros(1,4,2,2);cache=object()
    original=ChannelConcat([x],prepared=object(),second_cached=cache)
    result=ChannelConcat([x,original])
    assert result.prepared is None and result.second_cached is None
