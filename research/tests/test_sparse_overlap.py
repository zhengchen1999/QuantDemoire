"""Cross-stream sparse work must join before reading, including changing-input Graphs."""
import pytest
import torch
from helpers import synthetic_layer
from research.kernel import sparse_overlap
from kernel.ops import extension
from quantdemoire.precision import configure

pytestmark=[pytest.mark.experimental,pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')]

@pytest.mark.parametrize('bits',[4,8])
@pytest.mark.parametrize('warps',[0,8])
@pytest.mark.parametrize('geometry',[(1,19,17,1,1),(2,17,21,3,1),(1,9,7,2,2)])
@pytest.mark.parametrize('relu',[False,True])
@torch.inference_mode()
def test_sparse_overlap(bits,warps,geometry,relu):
    configure();torch.manual_seed(759)
    n,h,w,dilation,stride=geometry
    m=synthetic_layer(dict(weight=[32,96,3,3],stride=stride,padding=dilation,dilation=dilation),bits).cuda()
    m.main_half=True;m.joint_warps=warps;m.joint_variant=9;m.prepare_impl='quad_swizzle';m.spec['relu']=relu
    x=torch.randn(n,96,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    op=extension();s=m.spec
    def run(parallel=True):
        prepared=op.prepare_cached_cat([x],m.quant_multiplier,m.activation,bits,s['padded_channels'],m.sparse_channel_map,m.sparse_channels,5,True)
        return sparse_overlap.convolution(m,*prepared,parallel=parallel)
    assert torch.equal(run(),m(x));assert torch.equal(run(False),m(x))
    warm=torch.cuda.Stream();warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):run();run()
    torch.cuda.current_stream().wait_stream(warm)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):y=run()
    for scale in (3.,.3):
        x.normal_().mul_(scale);graph.replay();assert torch.equal(y,m(x))
