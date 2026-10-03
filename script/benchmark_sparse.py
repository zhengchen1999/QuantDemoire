"""Compare CSR-only kernels on actual weights and recorded ESDNet shapes."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import argparse
import json
from pathlib import Path
import torch
from kernel.ops import extension
from quantdemoire.deploy import load_model,PackedConv,sha256
from quantdemoire.esdnet import DenseBlock,ResidualDenseBlock
from quantdemoire.experiment import timing,write_json
from quantdemoire.precision import configure,metadata


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--shapes-report',required=True,help='a benchmark_kernel JSON with real runtime input shapes')
    p.add_argument('--output',required=True);p.add_argument('--all',action='store_true')
    p.add_argument('--repeats',type=int,default=10)
    a=p.parse_args();configure();torch.manual_seed(779)
    model=load_model(a.model,'cuda');op=extension();next_layers={};residual_layers=set()
    for block in model.modules():
        if isinstance(block,DenseBlock):
            for i,wrapper in enumerate(block.conv_layers):
                producer=wrapper.conv[0]
                consumer=block.conv_layers[i+1].conv[0] if i+1<len(block.conv_layers) else block.conv_post.conv
                if producer.can_quantize_for(consumer):next_layers[id(producer)]=consumer
        if isinstance(block,ResidualDenseBlock):residual_layers.add(id(block.conv_post.conv))
    records=[row['shape'] for row in json.loads(Path(a.shapes_report).read_text())['rows']]
    largest=max(r['input'][2]*r['input'][3] for r in records)
    result=dict(model=a.model,model_sha256=sha256(Path(a.model)/'model.safetensors'),shapes_report=a.shapes_report,
                shapes_report_sha256=sha256(a.shapes_report),all_layers=a.all,rows=[],complete=False,
                note='CSR-only CUDA Graph timing including optional next-quantization/residual epilogue, excluding input preparation and integer convolution. Random input, deployed sparse weights.')
    for record in records:
        m=model.get_submodule(record['name'])
        if not isinstance(m,PackedConv):continue
        if not a.all and record['input'][2]*record['input'][3]!=largest:continue
        s=m.spec;assert s['shape']==record['weight']
        n,c,h,w=record['input'];k,_,r,_=s['shape']
        x=torch.randn(record['input'],device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        packed,planar=op.prepare_cached_cat([x],m.quant_multiplier,m.activation,s['bits'],s['padded_channels'],m.sparse_channel_map,m.sparse_channels,2,True)
        loc=m.compact_locations
        dense=op.fused_conv(packed,planar,m.packed_weight,m.output_scale,m.row,loc,m.values,m.epilogue_sums,m.bias,m.activation,s['bits'],s['padded_channels'],k,r,s['stride'],s['padding'],s['dilation'],False,False)
        offsets=(loc&4095).long()*h*w+((loc>>16)&15).long()*s['dilation']*w+((loc>>12)&15).long()*s['dilation']
        nxt=next_layers.get(id(m));residual=torch.randn_like(dense) if id(m) in residual_layers else None
        common=(planar,dense,m.row,loc,m.values,m.activation,s['bits'],r,s['stride'],s['padding'],s['dilation'],s.get('relu',False))
        def legacy():
            if nxt is not None:return op.finish_half_next(*common,nxt.quant_multiplier,nxt.activation,nxt.sparse_channel_map,nxt.spec['bits'],nxt.spec['padded_channels'],nxt.sparse_channels)
            if residual is not None:return [op.finish_half_residual(*common,residual)]
            return [op.finish_half(*common)]
        def tuned(mode):
            return op.finish_half_tuned(planar,dense,m.row,loc,offsets,m.values,m.activation,s['bits'],r,s['stride'],s['padding'],s['dilation'],s.get('relu',False),
                residual if residual is not None else dense,residual is not None,nxt.quant_multiplier if nxt is not None else m.activation,
                nxt.activation if nxt is not None else m.activation,nxt.sparse_channel_map if nxt is not None else m.row,
                nxt.spec['bits'] if nxt is not None else 0,nxt.spec['padded_channels'] if nxt is not None else 0,nxt.sparse_channels if nxt is not None else 0,mode)
        expected=legacy();row=dict(layer=record['name'],shape=record,next_quant=nxt is not None,residual=residual is not None,timings={},bitwise_equal={})
        row['timings']['legacy']=timing(legacy,warmup=2,repeats=a.repeats,graph=True,graph_batch=10)
        for mode,name in [(1,'linear'),(2,'lut'),(3,'wide'),(4,'linear32'),(5,'divmod')]:
            actual=tuned(mode);equal=[torch.equal(actual[0],expected[0])]
            if nxt is not None:
                cp=nxt.spec['padded_channels'];bits=nxt.spec['bits'];p_,q_=dense.shape[2:]
                equal.append(torch.equal(actual[1].view(n,p_,q_,cp*bits//8)[...,:k*bits//8],expected[1].view(n,p_,q_,cp*bits//8)[...,:k*bits//8]))
                selected=nxt.sparse_channel_map[:k];selected=selected[selected>=0].long()
                equal.append(torch.equal(actual[2][:,selected],expected[2][:,selected]))
            row['bitwise_equal'][name]=equal
            if not all(equal):raise AssertionError((record['name'],name,equal))
            row['timings'][name]=timing(lambda:tuned(mode),warmup=2,repeats=a.repeats,graph=True,graph_batch=10)
        result['rows'].append(row);result['environment']=metadata();write_json(a.output,result)
        print(record['name'],{name:round(t['median_ms'],4) for name,t in row['timings'].items()},flush=True)
        del x,packed,planar,dense,expected,actual,residual
    result['complete']=True;result['layer_count']=len(result['rows'])
    result['sum_median_ms']={name:sum(row['timings'][name]['median_ms'] for row in result['rows']) for name in ('legacy','linear','lut','wide','linear32','divmod')}
    write_json(a.output,result);print(result['sum_median_ms'],flush=True)


if __name__=='__main__':main()
