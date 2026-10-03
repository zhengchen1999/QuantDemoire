"""Native FewChannels flag: small spatial dispatch plus larger-channel/pointwise fallbacks."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import itertools,sys
from pathlib import Path
import torch
from kernel import ops
from quantdemoire.deploy import balance_csr_rows
from quantdemoire.experiment import write_json
from quantdemoire.precision import configure
sys.path.insert(0,str(Path('tests').resolve()))
from helpers import synthetic_layer

@torch.inference_mode()
def test_few_channels_native(monkeypatch):
 configure();native=ops.extension();rows=[]
 class NativeFlags:
  def __getattr__(self,name):
   fn=getattr(native,name)
   if name not in ('joint_conv','joint_next'):return fn
   def dispatch(*args,**kwargs):
    args=list(args);enabled=args[23]==29
    if enabled:args[23]=8
    position=26 if name=='joint_conv' else 34
    if len(args)==position:args.append(enabled)
    else:args[position]=enabled
    return fn(*args,**kwargs)
   return dispatch
 op=NativeFlags();monkeypatch.setattr(ops,'extension',lambda:op)
 shapes=[(48,12,5,1,1,17,19),(36,48,3,2,3,17,19),(32,80,3,1,1,17,19),(64,144,1,1,1,9,11),(12,64,3,1,1,1,1),(96,80,3,1,2,13,21)]
 shapes += [(32,c,3,1,1,7,9) for c in (16,32,63,65,96,127,129,192,255,257,384,513)]
 for bits,shape,balanced in itertools.product((4,8),shapes,(False,True)):
  torch.manual_seed(813);k,c,r,stride,dil,h,w=shape;pad=(r//2)*dil
  m=synthetic_layer(dict(weight=[k,c,r,r],stride=stride,padding=pad,dilation=dil),bits).cuda()
  x=torch.randn(2,c,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
  row=balance_csr_rows(m.row) if balanced else m.row;off=m.sparse_linear_offsets(m.compact_locations,h,w,dil)
  p=(h+2*pad-dil*(r-1)-1)//stride+1;q=(w+2*pad-dil*(r-1)-1)//stride+1
  residual=torch.randn(2,k,p,q,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
  def call(variant,sparse=True,add=False):
   pk,pl=op.prepare_cached_cat([x],m.quant_multiplier,m.activation,bits,m.spec['padded_channels'],m.sparse_channel_map,m.sparse_channels,5,True)
   return op.joint_conv(pk,pl,m.packed_weight,m.output_scale,row,m.compact_locations,off,m.values,m.epilogue_sums,m.bias,m.activation,residual,
    bits,m.spec['padded_channels'],k,r,stride,pad,dil,True,sparse,add,8,variant,balanced,False)
  for sparse,add in itertools.product((False,True),(False,True)):
   expected=call(8,sparse,add);actual=call(29,sparse,add)
   assert torch.equal(expected,actual),(bits,shape,balanced,sparse,add,float((expected-actual).abs().max()))
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):call(29);call(29)
  torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):captured=call(29,True,True)
  for scale in (.1,50.):
   x.normal_().mul_(scale);residual.normal_().mul_(scale);expected=call(8,True,True);graph.replay();assert torch.equal(expected,captured),(bits,shape,balanced,'graph')
  rows.append(dict(bits=bits,shape=shape,balanced=balanced,equal=True,graph=True))
  print('conv',bits,shape,balanced,'passed',flush=True)
 # Reuse production prefix oracle, directing only spatial joint calls to FewChannels.
 class FewOps:
  def __getattr__(self,name):
   fn=getattr(op,name)
   if name not in ('joint_conv','joint_next'):return fn
   def dispatch(*args,**kwargs):
    args=list(args)
    if args[15]>1 and args[22]==8:args[23]=29
    return fn(*args,**kwargs)
   return dispatch
 monkeypatch.setattr(ops,'extension',lambda:FewOps())
 import test_joint_next as prefix
 monkeypatch.setattr(prefix,'extension',lambda:FewOps())
 # This module imported extension after replacement, so all native calls use this library.
 for bits,next_bits,shared,geometry,strategy,prefixed,balanced in itertools.product((4,8),(4,8),(False,True),((2,17,19,2),(1,1,1,1)),(0,1),(False,True),(False,True)):
  monkeypatch.setenv('QUANTDEMOIRE_BALANCE_CSR',str(int(balanced)))
  prefix.test_joint_next(bits,next_bits,3,shared,geometry,strategy,prefixed)
  rows.append(dict(kind='next',bits=bits,next_bits=next_bits,shared=shared,geometry=geometry,strategy=strategy,prefixed=prefixed,balanced=balanced,equal=True,graph=True))
 write_json('results/few_native_tests.json',dict(complete=True,rows=rows));print(len(rows),'cases passed',flush=True)
