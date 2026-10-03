"""Check every deployed quantizer's real concat signature and compare prepare kernels."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import argparse
from pathlib import Path
import torch
from quantdemoire.deploy import load_model,PackedConv,ChannelConcat,sha256
from quantdemoire.experiment import pad_image,timing,write_json
from quantdemoire.precision import configure,metadata
from kernel.ops import extension


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--height',type=int,default=2160)
    p.add_argument('--width',type=int,default=3840)
    p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--output',required=True)
    p.add_argument('--implementations',nargs='+',choices=['scalar','cached','pair','quad','quad_loop','quad_swizzle'],default=['scalar','cached','pair','quad','quad_loop','quad_swizzle'])
    a=p.parse_args();configure();torch.manual_seed(123)
    model=load_model(a.model,'cuda');records=[];handles=[]
    for name,m in model.named_modules():
        if isinstance(m,PackedConv):
            def record(module,inputs,name=name):
                x=inputs[0];parts=x.parts if isinstance(x,ChannelConcat) else [x]
                records.append(dict(name=name,parts=[list(part.shape) for part in parts]))
            handles.append(m.register_forward_pre_hook(record))
    x,_=pad_image(torch.rand(1,3,a.height,a.width,device='cuda',dtype=torch.float16))
    model(x.contiguous(memory_format=torch.channels_last))
    for handle in handles:handle.remove()
    del x
    assert len(records)==157
    op=extension();rows=[]
    for i,record in enumerate(records):
        m=model.get_submodule(record['name']);s=m.spec
        parts=[torch.randn(shape,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last) for shape in record['parts']]
        def scalar():return op.prepare_compact_cat(parts,m.inverse_smooth,m.activation,s['bits'],s['padded_channels'],m.sparse_channel_map,m.sparse_channels)
        def optimized(mode):return op.prepare_cached_cat(parts,m.quant_multiplier,m.activation,s['bits'],s['padded_channels'],m.sparse_channel_map,m.sparse_channels,mode,True)
        reference=scalar();actual=None;row=dict(**record,bits=s['bits'],sparse_channels=m.sparse_channels,bitwise_equal={})
        for name,mode in [('cached',1),('pair',2),('quad',3),('quad_loop',4),('quad_swizzle',5)]:
            if name not in a.implementations:continue
            actual=optimized(mode)
            row['bitwise_equal'][name]=all(torch.equal(x,y) for x,y in zip(reference,actual))
            if not row['bitwise_equal'][name]:raise AssertionError(f'quantized codes differ: {record["name"]}, {name}')
        del reference,actual
        calls=dict(scalar=scalar,cached=lambda:optimized(1),pair=lambda:optimized(2),quad=lambda:optimized(3),quad_loop=lambda:optimized(4),quad_swizzle=lambda:optimized(5))
        order=a.implementations if i%2==0 else list(reversed(a.implementations))
        row['timing']={name:timing(calls[name],warmup=2,repeats=a.repeats,graph=True,graph_batch=10) for name in order}
        rows.append(row)
        write_json(a.output,dict(environment=metadata(),model=a.model,weights_sha256=sha256(Path(a.model)/'model.safetensors'),count=len(rows),complete=len(rows)==157,
            sums_ms={name:sum(r['timing'][name]['median_ms'] for r in rows) for name in a.implementations},implementations=a.implementations,rows=rows))
        print(i,record['name'],{name:round(v['median_ms'],5) for name,v in row['timing'].items()},flush=True)


if __name__=='__main__':main()
