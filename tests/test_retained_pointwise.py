"""Check one/two A stages, tail channels, prefix layouts and changing Graph inputs."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import itertools
import pytest
from pathlib import Path
import torch
from kernel.ops import extension
from quantdemoire.deploy import balance_csr_rows
from quantdemoire.precision import configure
from helpers import synthetic_layer

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@torch.inference_mode()
def test_retained_pointwise_boundaries_and_prefixes():
 configure();op=extension();rows=[]
 cases=[(b,c) for b,cs in ((4,(48,240,256,272,352,496,512,528,576)),(8,(48,112,128,144,224,240,256,272,288))) for c in cs]
 for (bits,c),k,geometry,balanced in itertools.product(cases,(12,36),((2,17,19),(1,1,1)),(False,True)):
  torch.manual_seed(873);n,h,w=geometry
  m=synthetic_layer(dict(weight=[k,c,1,1],stride=1,padding=0,dilation=1),bits).cuda()
  x=torch.randn(n,c,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
  row=balance_csr_rows(m.row) if balanced else m.row;off=m.sparse_linear_offsets(m.compact_locations,h,w,1)
  def call(variant,sparse=True,residual=None):
   shared=(variant==21 and m.spec['padded_channels']<=(512 if bits==4 else 256)) or (variant==9 and m.spec['padded_channels']<=(256 if bits==4 else 128))
   pk,pl=op.prepare_cached_cat([x],m.quant_multiplier,m.activation,bits,m.spec['padded_channels'],m.sparse_channel_map,m.sparse_channels,5,not shared)
   return op.joint_conv(pk,pl,m.packed_weight,m.output_scale,row,m.locations if shared else m.compact_locations,off,m.values,m.epilogue_sums,m.bias,m.activation,residual if residual is not None else m.values,
    bits,m.spec['padded_channels'],k,1,1,0,1,True,sparse,residual is not None,8,(9 if variant==21 else variant),balanced,variant==21)
  residual=torch.randn(n,k,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
  for sparse in (False,True):assert torch.equal(call(9,sparse),call(21,sparse)),(bits,c,k,geometry,balanced,sparse)
  assert torch.equal(call(9,True,residual),call(21,True,residual))
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):call(21);call(21)
  torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):captured=call(21,True,residual)
  for scale in (.1,50.):
   x.normal_().mul_(scale);residual.normal_().mul_(scale);expected=call(9,True,residual);graph.replay();assert torch.equal(captured,expected),(bits,c,k,geometry,balanced,'graph')
  rows.append(dict(bits=bits,channels=c,k=k,geometry=geometry,balanced=balanced,equal=True,graph=True))
 print(len(rows),'one/two-stage output cases passed',flush=True)
 for bits,c,next_bits,strategy,balanced,planar in itertools.product((4,8),(144,240),(4,8),(0,1),(False,True),(False,True)):
  n,h,w,k=2,17,19,32
  m=synthetic_layer(dict(weight=[k,c,1,1],stride=1,padding=0,dilation=1),bits).cuda()
  nxt=synthetic_layer(dict(weight=[32,c+k,1,1],stride=1,padding=0,dilation=1),next_bits).cuda()
  x=torch.randn(n,c,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
  row=balance_csr_rows(m.row) if balanced else m.row;off=m.sparse_linear_offsets(m.compact_locations,h,w,1)
  def call(variant):
   shared=variant==21 or m.spec['padded_channels']<=(256 if bits==4 else 128)
   pk,pl=op.prepare_cached_cat([x],m.quant_multiplier,m.activation,bits,m.spec['padded_channels'],m.sparse_channel_map,m.sparse_channels,5,not shared)
   y,packed,aux=op.joint_next(pk,pl,m.packed_weight,m.output_scale,row,m.locations if shared else m.compact_locations,off,m.values,m.epilogue_sums,m.bias,m.activation,m.values,
    bits,m.spec['padded_channels'],k,1,1,0,1,True,True,False,8,(9 if variant==21 else variant),nxt.quant_multiplier,nxt.activation,nxt.sparse_channel_map,next_bits,nxt.spec['padded_channels'],nxt.sparse_channels,planar,strategy,balanced,variant==21)
   packed,aux=op.prepare_next_cat([y,x],nxt.quant_multiplier,nxt.activation,next_bits,nxt.spec['padded_channels'],nxt.sparse_channel_map,nxt.sparse_channels,packed,aux,k,5,planar)
   return (y,packed,aux) if planar else (y,packed)
  assert all(torch.equal(u,v) for u,v in zip(call(9),call(21)))
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):call(21);call(21)
  torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):captured=call(21)
  for scale in (.1,50.):
   x.normal_().mul_(scale);expected=call(9);graph.replay();assert all(torch.equal(u,v) for u,v in zip(expected,captured)),(bits,c,next_bits,strategy,balanced,planar,'prefix')
  rows.append(dict(bits=bits,channels=c,next_bits=next_bits,strategy=strategy,balanced=balanced,planar=planar,equal=True,graph=True))
 assert len(rows)==208
