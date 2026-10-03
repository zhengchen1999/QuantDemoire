"""Independent FP64 sparse-convolution check of the experimental MMA branch."""
import pytest
import torch
from torch.nn import functional as F
from helpers import synthetic_layer
from kernel.ops import extension
from research.kernel import sparse_tc_coalesced as candidate
from quantdemoire.branch_diagnostic import dense_core
from quantdemoire.precision import configure

pytestmark=[pytest.mark.experimental,pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')]

@pytest.mark.parametrize('bits',[4,8])
@pytest.mark.parametrize('geometry',[(1,19,17,1,1),(2,17,21,3,1),(1,9,7,2,2)])
@pytest.mark.parametrize('relu',[False,True])
@torch.inference_mode()
def test_mma_branch(bits,geometry,relu):
    configure();torch.manual_seed(437)
    n,h,w,dilation,stride=geometry
    m=synthetic_layer(dict(weight=[32,96,3,3],stride=stride,padding=dilation,dilation=dilation),bits).cuda()
    m.main_half=True;m.joint_warps=8;m.joint_variant=9;m.prepare_impl='quad_swizzle';m.spec['relu']=relu
    x=torch.randn(n,96,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    op=extension();s=m.spec
    candidate.prepare_metadata(m)
    def prepare():return op.prepare_cached_cat([x],m.quant_multiplier,m.activation,bits,s['padded_channels'],m.sparse_channel_map,m.sparse_channels,5,True)
    def run():return candidate.convolution(m,*prepare())
    def reference():
        packed,planar=prepare();s['relu']=False
        try:dense=dense_core(m,packed,planar,m.compact_locations)
        finally:s['relu']=relu
        a,z=m.activation;center=1<<(bits-1)
        code=(x.float()*m.quant_multiplier[None,:,None,None]+z).round().clamp(0,2**bits-1)
        activation=((code-center+(center-z))*a).half().double()
        weights=torch.zeros(32,96,3,3,device='cuda',dtype=torch.float64)
        row=m.row.cpu().tolist();locations=m.locations.cpu().tolist()
        for oc in range(32):
            for j in range(row[oc],row[oc+1]):
                loc=locations[j];weights[oc,loc&4095,(loc>>16)&15,(loc>>12)&15]=m.values[j].double()
        output=dense.double()+F.conv2d(activation,weights,None,stride,dilation,dilation)
        return (output.relu() if relu else output).half()
    def check(y):
        assert torch.isfinite(y).all()
        torch.testing.assert_close(y,reference(),atol=0.0005,rtol=0.001)
    check(run())
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):run();run()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):y=run()
    x.normal_().mul_(2);graph.replay();check(y)
