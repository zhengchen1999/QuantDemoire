"""Verify all full-network outputs when changing only activation preparation."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from quantdemoire.deploy import load_model,PackedConv,ChannelConcat,sha256
from quantdemoire.experiment import pad_image,write_json
from quantdemoire.precision import configure,metadata
from quantdemoire.esdnet import DenseBlock


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--data',default='data/UHDM/test');p.add_argument('--output',required=True)
    p.add_argument('--reference',choices=['scalar','cached','pair','quad','quad_loop','quad_swizzle'],default='scalar')
    p.add_argument('--candidate',choices=['scalar','cached','pair','quad','quad_loop','quad_swizzle'],default='pair')
    p.add_argument('--limit',type=int)
    p.add_argument('--next-quant',action='store_true',help='compare producer fusion off/on using the selected candidate prepare implementation')
    p.add_argument('--check-prefixes',action='store_true',help='also compare packed/planar buffers on every fused edge of the first image')
    p.add_argument('--graph',action='store_true',help='also verify candidate CUDA Graph replay while changing images')
    p.add_argument('--runtime-profile',action='store_true',help='apply the model folder runtime.json before loading')
    a=p.parse_args()
    if a.runtime_profile:
        from quantdemoire.profiles import apply_execution_profile
        apply_execution_profile(a.model)
    configure();torch.backends.cudnn.benchmark=False
    model=load_model(a.model,'cuda');layers=[m for m in model.modules() if isinstance(m,PackedConv)]
    blocks=[m for m in model.modules() if isinstance(m,DenseBlock)]
    if a.next_quant and (a.candidate not in ('pair','quad','quad_loop','quad_swizzle') or not all(m.supports_prepared_input() for m in layers)):
        raise ValueError('next-quant verification requires pair/quad preparation and compact CSR / half main output')
    paths=sorted(Path(a.data).rglob('*_moire.jpg'))
    if a.limit:paths=paths[:a.limit]
    if not paths:raise ValueError('empty test split')
    result=dict(model=a.model,model_sha256=sha256(Path(a.model)/'model.safetensors'),
                config_sha256=sha256(Path(a.model)/'config.json'),reference=a.reference,candidate=a.candidate,next_quant=a.next_quant,graph=a.graph,rows=[])
    runner=None
    if a.check_prefixes and not a.next_quant:raise ValueError('--check-prefixes requires --next-quant')
    handles=[];prefix_rows=[];reference_features={};capture_reference=False
    if a.check_prefixes:
        from kernel.ops import extension
        producer_by_consumer={};sources_by_consumer={}
        module_names={id(m):name for name,m in model.named_modules()}
        for block in blocks:
            chain=[layer.conv[0] for layer in block.conv_layers]+[block.conv_post.conv]
            for producer,consumer in zip(chain,chain[1:]):producer_by_consumer[id(consumer)]=producer
            for index,consumer in enumerate(chain):sources_by_consumer[id(consumer)]=list(reversed(chain[:index]))
            # An omitted feature has no FP16 data to quantize as an oracle.
            # Save the unfused reference's final two features on CPU instead.
            for producer in chain[-3:-1]:
                def save_reference(m,args,value):
                    if capture_reference:reference_features[id(m)]=value.detach().cpu()
                handles.append(producer.register_forward_hook(save_reference))
        for name,module in model.named_modules():
            if not isinstance(module,PackedConv):continue
            def check_prefix(m,args,name=name):
                x=args[0]
                if not isinstance(x,ChannelConcat) or x.prepared is None:return
                _,packed,planar,skip=x.prepared;s=m.spec;op=extension()
                reference_parts=[];omitted_segments=[];omitted_sources=[]
                for index,part in enumerate(x.parts):
                    if getattr(part,'_quantdemoire_omitted',False):
                        source=sources_by_consumer[id(m)][index]
                        part=reference_features[id(source)].to(device=part.device,memory_format=torch.channels_last)
                        omitted_segments.append(index)
                        omitted_sources.append(module_names[id(source)])
                    reference_parts.append(part)
                expected=op.prepare_cached_cat(reference_parts,m.quant_multiplier,m.activation,s['bits'],s['padded_channels'],m.sparse_channel_map,m.sparse_channels,2,True)
                needs_planar=not (m.shared_skip_planar and m.uses_shared_pointwise())
                if x.second_cached is not None:
                    actual=op.prepare_cached_second(x.parts,m.quant_multiplier,m.activation,s['bits'],s['padded_channels'],m.sparse_channel_map,m.sparse_channels,packed,planar,skip,needs_planar,x.second_cached)
                else:
                    actual=op.prepare_next_cat(x.parts,m.quant_multiplier,m.activation,s['bits'],s['padded_channels'],m.sparse_channel_map,m.sparse_channels,packed,planar,skip,{'pair':2,'quad':3,'quad_loop':4,'quad_swizzle':5}[m.prepare_impl],needs_planar)
                packed_equal=torch.equal(actual[0],expected[0])
                planar_equal=torch.equal(actual[1],expected[1]) if needs_planar else None
                producer=producer_by_consumer[id(m)]
                prefix_rows.append(dict(consumer=name,producer=module_names[id(producer)],producer_warps=producer.joint_warps,
                    joint_next_mode=producer.joint_next_mode,input=list(x.shape),prefix_channels=skip,packed_bitwise_equal=packed_equal,
                    planar_bitwise_equal=planar_equal,planar_checked=needs_planar,planar_storage_bytes=actual[1].untyped_storage().nbytes(),
                    second_cache_bytes=x.second_cached.untyped_storage().nbytes() if x.second_cached is not None else 0,
                    omitted_segments=omitted_segments,omitted_sources=omitted_sources,
                    omitted_reference='unfused producer features retained on CPU' if omitted_segments else None))
                if not packed_equal or (needs_planar and not planar_equal):raise AssertionError(f'prepared input changed: {name}')
            handles.append(module.register_forward_pre_hook(check_prefix))
    for i,path in enumerate(paths):
        with Image.open(path) as im:
            x=torch.from_numpy(np.asarray(im.convert('RGB'),dtype=np.float32)/255).permute(2,0,1)[None].cuda()
        x,_=pad_image(x);x=x.half().contiguous(memory_format=torch.channels_last)
        for m in layers:m.prepare_impl=a.reference
        for b in blocks:b.fuse_next_quant=False
        capture_reference=(i==0 and a.check_prefixes)
        expected=model(x)
        capture_reference=False
        for m in layers:m.prepare_impl=a.candidate
        for b in blocks:b.fuse_next_quant=a.next_quant
        actual=model(x)
        if i==0 and a.check_prefixes:
            for handle in handles:handle.remove()
            reference_features.clear()
            if not prefix_rows:raise AssertionError('no producer fusion exercised')
            result['prefix_checks']=prefix_rows
        equal=[torch.equal(x,y) for x,y in zip(actual,expected)]
        row=dict(image=str(path.relative_to(a.data)),input_sha256=sha256(path),bitwise_equal=equal)
        if not all(equal):row['max_abs']=[float((x-y).abs().max()) for x,y in zip(actual,expected)]
        if a.graph:
            if runner is None:
                from quantdemoire.runtime import CudaGraphRunner
                runner=CudaGraphRunner(model,x)
            captured=runner(x)
            row['graph_bitwise_equal']=[torch.equal(u,v) for u,v in zip(expected,captured)]
            if not all(row['graph_bitwise_equal']):raise AssertionError(f'graph output changed: {row}')
        result['rows'].append(row);result['count']=len(result['rows']);result['complete']=result['count']==len(paths)
        result['environment']=dict(metadata(),fuse_next_quant=a.next_quant);write_json(a.output,result)
        if not all(equal):raise AssertionError(f'output changed: {row}')
        if (i+1)%10==0:print(i+1,'all three outputs bitwise equal',flush=True)
        del x,actual,expected


if __name__=='__main__':main()
