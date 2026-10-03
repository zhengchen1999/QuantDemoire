"""Two distinct consumer quantizers must preserve output, prefix, and future codes."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import itertools,sys
from pathlib import Path
import torch
from kernel.ops import extension as reference_extension
from quantdemoire.experiment import write_json
from quantdemoire.precision import configure
from quantdemoire.deploy import balance_csr_rows
import pytest
from types import SimpleNamespace
def extension():return SimpleNamespace(joint_next=reference_extension().joint_next_two)
def arguments(native,producer,parts,consumer):
    m=producer;s=m.spec
    n,_,h,w=parts[0].shape;k,_,r,_=s['shape']
    assert m.joint_warps==8 and m.joint_next_mode>0 and m.balance_csr and m.packed_csr_meta
    assert r>1 and not m.uses_shared_pointwise()
    packed,planar=native.prepare_cached_cat(parts,m.quant_multiplier,m.activation,s['bits'],
        s['padded_channels'],m.sparse_channel_map,m.sparse_channels,5,True)
    offsets=m.sparse_linear_offsets(m.compact_locations,h,w,s['dilation'],packed=True)
    nxt=consumer
    return (packed,planar,m.packed_weight,m.output_scale,m.balanced_row,m.compact_locations,
        offsets,m.values,m.epilogue_sums,m.bias,m.activation,m.values,s['bits'],s['padded_channels'],
        k,r,s['stride'],s['padding'],s['dilation'],s.get('relu',False),True,False,8,m.joint_variant,
        nxt.quant_multiplier,nxt.activation,nxt.sparse_channel_map,nxt.spec['bits'],
        nxt.spec['padded_channels'],nxt.sparse_channels,
        not (nxt.shared_skip_planar and nxt.shared_skip_prefix_planar and nxt.uses_shared_pointwise()),
        m.joint_next_mode-1,True,m.retain_shared,m.few_channels,True)

def future_arguments(consumer,offset):
    return (consumer.quant_multiplier,consumer.activation,consumer.spec['bits'],offset)

from helpers import synthetic_layer

@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@torch.inference_mode()
def test_native():
    configure();native=reference_extension();candidate=extension();rows=[]
    cases=itertools.product((4,8),(4,8),(4,8),(48,80),(0,1),(False,True))
    for bits,next_bits,future_bits,c,strategy,next_planar in cases:
        torch.manual_seed(609)
        n,h,w,dil=(2,17,19,2) if c==48 else (1,1,1,1)
        m=synthetic_layer(dict(weight=[32,c,3,3],stride=1,padding=dil,dilation=dil),bits).cuda()
        nxt=synthetic_layer(dict(weight=[32,c+32,1 if not next_planar else 3,1 if not next_planar else 3],stride=1,padding=0 if not next_planar else 1,dilation=1),next_bits).cuda()
        future=synthetic_layer(dict(weight=[32,c+64,3,3],stride=1,padding=1,dilation=1),future_bits).cuda()
        for layer in (m,nxt,future):
            layer.main_half=True;layer.sparse_enabled=True;layer.compact_sparse=True
            layer.prepare_impl='quad_swizzle';layer.joint_warps=8;layer.joint_variant=9
        m.joint_next_mode=strategy+1;m.balance_csr=True;m.packed_csr_meta=True;m.few_channels=True
        m.balanced_row=balance_csr_rows(m.row)
        m.spec['relu']=c==48
        nxt.shared_skip_planar=not next_planar;nxt.shared_skip_prefix_planar=not next_planar
        x=torch.randn(n,c,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        mapping=torch.arange(32,device='cuda',dtype=torch.int32)
        def call():
            args=arguments(native,m,[x],nxt)
            expected=native.joint_next(*args)
            actual=candidate.joint_next(*args,*future_arguments(future,32))
            # Only the produced prefix is initialized; completing the input permits
            # an exact packed/planar comparison without reading uninitialized bytes.
            def finish(result):
                return native.prepare_next_cat([result[0],x],nxt.quant_multiplier,nxt.activation,next_bits,
                    nxt.spec['padded_channels'],nxt.sparse_channel_map,nxt.sparse_channels,result[1],result[2],32,5,next_planar)
            reference_completed=finish(expected);actual_completed=finish(actual)
            future_expected=native.prepare_cached_cat([actual[0]],future.quant_multiplier[32:64],
                future.activation,future_bits,32,mapping,32,5,False)[0]
            return (expected[0],actual[0],*reference_completed,*actual_completed,future_expected,actual[3])
        def validate(v):
            assert torch.equal(v[0],v[1]),'producer output'
            assert torch.equal(v[2],v[4]),'next prefix'
            if next_planar:assert torch.equal(v[3],v[5]),'next planar'
            else:assert v[5].untyped_storage().nbytes()==1
            assert torch.equal(v[6],v[7]),'second successor'
        validate(call())
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):call();call()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=call()
        for scale in (.01,100.):
            x.normal_().mul_(scale);graph.replay();validate(captured)
        rows.append(dict(bits=bits,next_bits=next_bits,future_bits=future_bits,channels=c,
            strategy=strategy,next_planar=next_planar,equal=True,graph=True))
        print(len(rows),bits,next_bits,future_bits,c,strategy,next_planar,'passed',flush=True)
    write_json('results/second_producer_native_tests.json',dict(complete=True,rows=rows,note=__doc__))
