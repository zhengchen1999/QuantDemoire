"""Cached second feature must reproduce complete consumer-specific quantization."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import itertools,sys
from pathlib import Path
import torch
from kernel.ops import extension as reference_extension
from quantdemoire.precision import configure
from quantdemoire.experiment import write_json
import pytest
extension=reference_extension
from helpers import synthetic_layer

@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@torch.inference_mode()
def test_native():
    configure();native=reference_extension();op=extension();rows=[]
    shapes=[(2,17,19,[32,32,48]),(1,1,1,[16,32,12]),(2,7,1,[48,64,20]),
            (1,9,13,[32,32,32,48])]
    for bits,sparse,shape in itertools.product((4,8),(False,True),shapes):
        torch.manual_seed(492);n,h,w,channels=shape;c=sum(channels);skip=channels[0];cached=channels[1]
        layer=synthetic_layer(dict(weight=[36,c,3,3],stride=1,padding=1,dilation=1),bits).cuda()
        cp=layer.spec['padded_channels'];sc=layer.sparse_channels
        parts=[torch.randn(n,ch,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last) for ch in channels]
        cached_map=torch.arange(cached,device='cuda',dtype=torch.int32)
        targets=layer.sparse_channel_map[:skip];targets=targets[targets>=0].long()
        def call():
            expected=native.prepare_cached_cat(parts,layer.quant_multiplier,layer.activation,bits,cp,
                layer.sparse_channel_map,sc,5,sparse)
            second=native.prepare_cached_cat([parts[1]],layer.quant_multiplier[skip:skip+cached],
                layer.activation,bits,cached,cached_map,cached,5,False)[0]
            packed=torch.full_like(expected[0],0xa5)
            packed.view(-1,cp*bits//8)[:,:skip*bits//8].copy_(expected[0].view(-1,cp*bits//8)[:,:skip*bits//8])
            if sparse:
                planar=torch.full_like(expected[1],-91)
                planar.index_copy_(1,targets,expected[1].index_select(1,targets))
            else:planar=expected[1]
            actual=op.prepare_cached_second(parts,layer.quant_multiplier,layer.activation,bits,cp,
                layer.sparse_channel_map,sc,packed,planar,skip,sparse,second)
            return (*expected,*actual)
        def validate(values):
            pk,pl,actual_pk,actual_pl=values
            assert torch.equal(pk,actual_pk),(bits,sparse,shape,'packed')
            if sparse:assert torch.equal(pl,actual_pl),(bits,shape,'planar')
            else:assert actual_pl.untyped_storage().nbytes()==1
        validate(call())
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):call();call()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=call()
        for scale in (.01,100.):
            for part in parts:part.normal_().mul_(scale)
            graph.replay();validate(captured)
        rows.append(dict(bits=bits,sparse=sparse,shape=shape,equal=True,graph=True))
        print(bits,sparse,shape,'passed',flush=True)
    write_json('results/second_prepare_native_tests.json',dict(complete=True,rows=rows,note=__doc__))
