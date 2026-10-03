"""Compare separated/joint output kernels for every deployed ESDNet shape.

Preparation is excluded. This does not model the next-quant prefix benefit;
deployment decisions must additionally pass a complete-network benchmark.
"""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import argparse
import json
from pathlib import Path
import torch
from kernel.ops import extension
from quantdemoire.deploy import load_model,PackedConv,sha256
from quantdemoire.esdnet import ResidualDenseBlock
from quantdemoire.experiment import timing,write_json
from quantdemoire.precision import configure,metadata


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--shapes-report',required=True);p.add_argument('--output',required=True)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--runtime-profile',action='store_true',help='apply the model folder runtime.json before loading')
    a=p.parse_args()
    if a.runtime_profile:
        from quantdemoire.profiles import apply_execution_profile
        apply_execution_profile(a.model)
    configure();torch.manual_seed(194)
    model=load_model(a.model,'cuda');op=extension()
    residual_layers={id(m.conv_post.conv) for m in model.modules() if isinstance(m,ResidualDenseBlock)}
    records=[r['shape'] for r in json.loads(Path(a.shapes_report).read_text())['rows']]
    result=dict(model=a.model,model_sha256=sha256(Path(a.model)/'model.safetensors'),
                shapes_report=a.shapes_report,shapes_report_sha256=sha256(a.shapes_report),
                rows=[],complete=False,note=__doc__)
    for record in records:
        m=model.get_submodule(record['name'])
        if not isinstance(m,PackedConv):continue
        s=m.spec;n,c,h,w=record['input'];k,_,r,_=s['shape']
        x=torch.randn(record['input'],device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        packed,planar=op.prepare_cached_cat([x],m.quant_multiplier,m.activation,s['bits'],s['padded_channels'],m.sparse_channel_map,m.sparse_channels,{'cached':1,'pair':2,'quad':3,'quad_loop':4,'quad_swizzle':5}.get(m.prepare_impl,2),True)
        omit_planar=m.shared_skip_planar and m.joint_variant==9 and r==1 and s['stride']==1 and s['padding']==0 and s['padded_channels']<=((512 if s['bits']==4 else 256) if m.retain_shared else (256 if s['bits']==4 else 128))
        mock_planar=torch.empty(1,device=planar.device,dtype=planar.dtype).as_strided(planar.shape,(0,0,0,0)) if omit_planar else planar
        loc=m.compact_locations;offsets=m.sparse_linear_offsets(loc,h,w,s['dilation'])
        shared_geometry=m.joint_variant==9 and r==1 and s['stride']==1 and s['padding']==0 and s['padded_channels']<=((512 if s['bits']==4 else 256) if m.retain_shared else (256 if s['bits']==4 else 128))
        metadata_eligible=m.packed_csr_meta and m.joint_variant in (8,9) and not shared_geometry and planar.numel()<2147483647
        packed_offsets=m.sparse_linear_offsets(loc,h,w,s['dilation'],packed=True) if metadata_eligible else offsets
        def dense():
            return op.fused_conv(packed,planar,m.packed_weight,m.output_scale,m.row,loc,m.values,m.epilogue_sums,m.bias,m.activation,
                s['bits'],s['padded_channels'],k,r,s['stride'],s['padding'],s['dilation'],False,False)
        base=dense();residual=torch.randn_like(base) if id(m) in residual_layers else None
        def separate():
            y=dense()
            return op.finish_half_tuned(planar,y,m.row,loc,offsets,m.values,m.activation,s['bits'],r,s['stride'],s['padding'],s['dilation'],s.get('relu',False),
                residual if residual is not None else y,residual is not None,m.activation,m.activation,m.row,0,0,0,{'linear':1,'lut':2,'wide':3,'linear32':4,'divmod':5}.get(m.csr_kernel,1))[0]
        def joint(warps,sparse=True):
            shared=m.joint_variant==9 and warps==8 and r==1 and s['stride']==1 and s['padding']==0 and s['padded_channels']<=((512 if s['bits']==4 else 256) if m.retain_shared else (256 if s['bits']==4 else 128))
            joint_loc=m.locations if shared else loc
            packed_meta=metadata_eligible and warps==8 and sparse
            balanced=m.balance_csr and warps==8 and m.joint_variant in (8,9)
            return op.joint_conv(packed,mock_planar if shared else planar,m.packed_weight,m.output_scale,m.balanced_row if balanced else m.row,joint_loc,packed_offsets if packed_meta else offsets,m.values,m.epilogue_sums,m.bias,m.activation,
                residual if residual is not None else m.values,s['bits'],s['padded_channels'],k,r,s['stride'],s['padding'],s['dilation'],s.get('relu',False),sparse,residual is not None,warps,m.joint_variant,balanced,m.retain_shared,m.few_channels,packed_meta)
        expected=separate();row=dict(layer=record['name'],shape=record,residual=residual is not None,shared_planar_omitted=omit_planar,packed_csr_meta=metadata_eligible,csr_balanced=m.balance_csr and m.joint_variant in (8,9),timings={},bitwise_equal={})
        calls={'separate':separate,'joint4':lambda:joint(4),'joint8':lambda:joint(8),'dense':dense,'joint8_dense':lambda:joint(8,False)}
        for name in ('joint4','joint8'):
            actual=calls[name]();row['bitwise_equal'][name]=torch.equal(actual,expected)
            if not row['bitwise_equal'][name]:raise AssertionError((record['name'],name,float((actual-expected).abs().max())))
        # Alternate order across layers to reduce a consistent ordering bias.
        names=list(calls) if len(result['rows'])%2==0 else list(reversed(calls))
        for name in names:row['timings'][name]=timing(calls[name],warmup=2,repeats=a.repeats,graph=True,graph_batch=10)
        row['joint8_sparse_increment_ratio']=row['timings']['joint8']['median_ms']/row['timings']['joint8_dense']['median_ms']-1
        result['rows'].append(row);result['environment']=metadata();write_json(a.output,result)
        print(record['name'],{name:round(t['median_ms'],4) for name,t in row['timings'].items()},flush=True)
        del x,packed,planar,mock_planar,base,residual,expected,actual,offsets,packed_offsets
    result['complete']=True;result['layer_count']=len(result['rows'])
    result['sum_median_ms']={name:sum(row['timings'][name]['median_ms'] for row in result['rows']) for name in calls}
    result['joint8_sparse_increment_ratio']=result['sum_median_ms']['joint8']/result['sum_median_ms']['joint8_dense']-1
    write_json(a.output,result);print(result['sum_median_ms'],flush=True)


if __name__=='__main__':main()
