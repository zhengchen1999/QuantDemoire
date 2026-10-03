"""Screen experimental spatial CSR kernels on actual weights and hotspot shapes.

Inputs are synthetic. This screen excludes consumer caches and is not an
end-to-end speed or branch-budget acceptance test.
"""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import argparse
import json
from pathlib import Path
import statistics
import importlib
import hashlib
import torch
from kernel.ops import extension
from quantdemoire.deploy import load_model, sha256
from quantdemoire.experiment import timing, write_json
from quantdemoire.precision import configure, metadata
from quantdemoire.profiles import apply_execution_profile


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--branch-report',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--repeats',type=int,default=30)
    parser.add_argument('--candidate',choices=('spatial_packed','sparse_prefetch','sparse_products','sparse_half2','planar_int4','sparse_subwarp','sparse_tc_coalesced','sparse_overlap'),default='spatial_packed')
    args=parser.parse_args()
    configure();torch.manual_seed(369);apply_execution_profile(args.model,overwrite=True)
    candidate_module=importlib.import_module('research.kernel.'+args.candidate)
    model=load_model(args.model,'cuda');op=extension();compiled=candidate_module.extension()
    baseline=json.loads(Path(args.branch_report).read_text())
    weight_hash=sha256(Path(args.model)/'model.safetensors')
    if baseline['model_sha256']!=weight_hash:
        raise ValueError('branch report belongs to other weights')
    selected=sorted((r for r in baseline['rows'] if r['shape'][2]>1 and r['shape'][0]%8==0),
                    key=lambda r:-r['incremental_ms'])[:args.count]
    report=dict(complete=False,note=__doc__,candidate=args.candidate,weights_sha256=weight_hash,
        candidate_binary_sha256=sha256(compiled.__file__),
        candidate_source_sha256=sha256(candidate_module.__file__),
        generated_sources_sha256=hashlib.sha256(b''.join(name.encode()+value.encode()
            for name,value in sorted(candidate_module.sources().items()))).hexdigest(),
        caller_source_sha256=sha256(Path('research/kernel/spatial_packed.py')),
        helper_source_sha256=sha256(Path('quantdemoire/branch_diagnostic.py')),
        metadata_cache='one representation per call; no repeated unpacked/packed rebuild',rows=[])
    for record in selected:
        m=model.get_submodule(record['layer']);s=m.spec
        x=torch.randn(record['input'],device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        def prepare(planar):
            return op.prepare_cached_cat([x],m.quant_multiplier,m.activation,s['bits'],s['padded_channels'],
                m.sparse_channel_map,m.sparse_channels,5,planar)
        def candidate():
            packed,planar=(candidate_module.prepare(m,[x]) if args.candidate=='planar_int4'
                           else prepare(args.candidate!='spatial_packed'))
            return candidate_module.convolution(m,packed,planar)
        expected=m(x);actual=candidate()
        equal=torch.equal(expected,actual)
        row=dict(layer=record['layer'],shape=record['shape'],input=record['input'],
            joint_warps=m.joint_warps,bitwise_equal=equal,max_abs=float((expected-actual).abs().max()),measurements=[])
        if not equal and args.candidate not in ('sparse_half2','sparse_tc_coalesced'):
            report['rows'].append(row);write_json(args.output,report)
            raise RuntimeError(f"candidate differs at {record['layer']}")
        if not torch.isfinite(actual).all():raise RuntimeError('candidate produced nonfinite output')
        for round_index in range(2):
            for mode in (('baseline','candidate') if round_index==0 else ('candidate','baseline')):
                result=timing((lambda:m(x)) if mode=='baseline' else candidate,
                    warmup=5,repeats=args.repeats,graph=True,graph_batch=3)
                row['measurements'].append(dict(round=round_index,path=mode,timing=result))
        row['median_ms']={mode:statistics.median(r['timing']['median_ms'] for r in row['measurements'] if r['path']==mode)
                          for mode in ('baseline','candidate')}
        row['speedup']=row['median_ms']['baseline']/row['median_ms']['candidate']
        row['prepare_planar']=timing(lambda:prepare(True),repeats=args.repeats,graph=True,graph_batch=3)
        row['prepare_packed_only']=timing(lambda:prepare(False),repeats=args.repeats,graph=True,graph_batch=3)
        if args.candidate=='sparse_overlap':
            def serial():return candidate_module.convolution(m,*prepare(True),parallel=False)
            row['serial_same_kernels']=timing(serial,warmup=5,repeats=args.repeats,graph=True,graph_batch=3)
        report['rows'].append(row);write_json(args.output,report)
        print(row['layer'],row['median_ms'],row['speedup'],flush=True)
    sums={mode:sum(row['median_ms'][mode] for row in report['rows']) for mode in ('baseline','candidate')}
    report.update(complete=True,environment=metadata(),sums_ms=sums,speedup=sums['baseline']/sums['candidate'])
    write_json(args.output,report)
    print(report['sums_ms'],report['speedup'])


if __name__=='__main__':main()
