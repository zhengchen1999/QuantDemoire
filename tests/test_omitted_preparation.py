"""Partial preparation must never read any wholly cached metadata-only segment."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import pytest
import torch
from kernel.ops import extension
from quantdemoire.precision import configure

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def placeholder(tensor):
    return torch.empty(1,device=tensor.device,dtype=tensor.dtype).as_strided(tensor.shape,(0,0,0,0))


@pytest.mark.parametrize('bits',[4,8])
@pytest.mark.parametrize('sparse',[False,True])
@pytest.mark.parametrize('geometry',[(2,17,19),(1,1,1),(1,8,16)])
@pytest.mark.parametrize('pattern',[(True,False,False),(True,False,True),(False,True,True),(True,True,True)])
@torch.inference_mode()
def test_partial_cached_segments(bits,sparse,geometry,pattern):
    configure();op=extension();n,h,w=geometry;omit_first,omit_second,has_second=pattern;torch.manual_seed(993)
    real=[torch.randn(n,c,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last) for c in (32,32,48)]
    inputs=[placeholder(t) if flag else t for t,flag in zip(real,(omit_first,omit_second,False))]
    multiplier=torch.rand(112,device='cuda')*2+.5
    params=torch.tensor([.125,11 if bits==4 else 113],device='cuda',dtype=torch.float32)
    cp=128 if bits==4 else 112
    used=torch.tensor([i for i in range(112) if i%3!=1],device='cuda',dtype=torch.long)
    selection=torch.full((112,),-1,device='cuda',dtype=torch.int32)
    selection[used]=torch.arange(used.numel(),device='cuda',dtype=torch.int32);sc=used.numel()
    prefix_targets=selection[:32][selection[:32]>=0].long();second_map=torch.arange(32,device='cuda',dtype=torch.int32)
    def call():
        expected=op.prepare_cached_cat(real,multiplier,params,bits,cp,selection,sc,5,sparse)
        packed=torch.full_like(expected[0],165)
        packed.view(-1,cp*bits//8)[:,:32*bits//8].copy_(expected[0].view(-1,cp*bits//8)[:,:32*bits//8])
        if sparse:
            planar=torch.full_like(expected[1],93)
            planar.index_copy_(1,prefix_targets,expected[1].index_select(1,prefix_targets))
        else:planar=torch.empty(1,device='cuda',dtype=torch.int8).as_strided(expected[1].shape,(0,0,0,0))
        if has_second:
            second=op.prepare_cached_cat([real[1]],multiplier[32:64],params,bits,32,second_map,32,5,False)[0]
            actual=op.prepare_cached_second(inputs,multiplier,params,bits,cp,selection,sc,packed,planar,32,sparse,second)
        else:actual=op.prepare_next_cat(inputs,multiplier,params,bits,cp,selection,sc,packed,planar,32,5,sparse)
        return expected,actual
    def validate(result):
        expected,actual=result;assert torch.equal(expected[0],actual[0])
        if sparse:assert torch.equal(expected[1],actual[1])
        else:assert actual[1].untyped_storage().nbytes()==1
    validate(call())
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):validate(call());call()
    torch.cuda.current_stream().wait_stream(stream)
    if os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING')!='1':
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=call()
        for magnitude in (.001,100.):
            for tensor in real:tensor.normal_().mul_(magnitude)
            graph.replay();validate(captured)
    expected,_=call()
    # An uncached suffix never receives the relaxed first/second layout check.
    bad=[inputs[0],real[1],placeholder(real[2])]
    with pytest.raises(RuntimeError,match='invalid concat operand'):
        op.prepare_next_cat(bad,multiplier,params,bits,cp,selection,sc,*expected,32,5,sparse)
    with pytest.raises(RuntimeError,match='input must be channels-last'):
        op.prepare_cached_cat([placeholder(real[0]),*real[1:]],multiplier,params,bits,cp,selection,sc,5,sparse)
    with pytest.raises(RuntimeError,match='input must be channels-last'):
        op.prepare_next_cat([placeholder(real[0]),*real[1:]],multiplier,params,bits,cp,selection,sc,*expected,32,2,sparse)
