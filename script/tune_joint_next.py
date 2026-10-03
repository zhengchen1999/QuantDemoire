"""Choose separate/joint execution and each joint producer's prefix strategy.

Costs include preparation, output processing, and consumer-specific prefix
generation. A producer's strategy determines the following input state; the
following layer's joint mode can determine whether auxiliary prefix data is
needed. Whole-network timing and validation remain mandatory after tuning.
"""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import torch
from kernel.ops import extension
from quantdemoire.deploy import load_model,PackedConv,ChannelConcat,sha256
from quantdemoire.esdnet import DenseBlock,ResidualDenseBlock
from quantdemoire.experiment import timing,write_json
from quantdemoire.precision import configure,metadata


MODES={'separate':(0,0,True),'joint':(8,0,False),
       'direct':(8,1,True),'transpose':(8,2,True)}


def solve_chain(chain,rows,relative_margin,absolute_margin,prefix_margin):
    @lru_cache(None)
    def visit(index,previous,current):
        kind='prefix' if previous is not None and MODES[previous][2] else 'full'
        following_modes=rows[chain[index+1]]['modes'] if index+1<len(chain) else (None,)
        candidates=[]
        for following in following_modes:
            key='none' if following is None else str(MODES[following][0])
            costs=rows[chain[index]]['timing'][kind]
            value=costs[current][key]['median_ms']
            if current!='separate':
                value+=max(absolute_margin,costs['separate'][key]['median_ms']*relative_margin)
            if current in ('direct','transpose'):value+=prefix_margin
            tail_score,tail=visit(index+1,current,following) if following is not None else (0.,())
            candidates.append((value+tail_score,(current,*tail)))
        return min(candidates)
    score,modes=min(visit(0,None,mode) for mode in rows[chain[0]]['modes'])
    return dict(zip(chain,modes)),score


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--shapes',required=True);p.add_argument('--output',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--repeats',type=int,default=8);p.add_argument('--relative-margin',type=float,default=.01)
    p.add_argument('--absolute-margin-ms',type=float,default=.001);p.add_argument('--prefix-margin-ms',type=float,default=.001)
    a=p.parse_args()
    if int(os.environ.get('QUANTDEMOIRE_JOINT_NEXT_QUANT','0')) not in (1,2):
        p.error('enable QUANTDEMOIRE_JOINT_NEXT_QUANT=1 or 2 before tuning')
    if min(a.relative_margin,a.absolute_margin_ms,a.prefix_margin_ms)<0 or a.repeats<1:p.error('invalid margins or repeat count')
    configure();torch.manual_seed(994)
    source=json.loads(Path(a.shapes).read_text());weight_hash=sha256(Path(a.model)/'model.safetensors')
    if not source['complete'] or source['count']!=157 or source['weights_sha256']!=weight_hash:
        raise ValueError('shape report does not cover this model')
    model=load_model(a.model,'cuda');op=extension()
    modules={name:m for name,m in model.named_modules() if isinstance(m,PackedConv)}
    records={row['name']:row for row in source['rows']}
    if set(modules)!=set(records):raise ValueError('model and shape names differ')
    if any(m.joint_variant not in (8,9) for m in modules.values()):raise ValueError('joint producer tuning requires variant 8 or 9')
    chains=[];successors={};predecessors={};residuals=set();assigned=set()
    for name,block in model.named_modules():
        if not isinstance(block,DenseBlock):continue
        chain=[f'{name}.conv_layers.{i}.conv.0' for i in range(len(block.conv_layers))]+[f'{name}.conv_post.conv']
        for left,right in zip(chain,chain[1:]):
            modules[left].joint_warps=8;modules[left].joint_next_mode=1
            if not modules[left].can_quantize_for(modules[right]):raise ValueError(f'unsupported joint edge: {left}')
            successors[left]=right;predecessors[right]=left
        if isinstance(block,ResidualDenseBlock):residuals.add(chain[-1])
        chains.append(chain);assigned.update(chain)
    chains += [[name] for name in modules if name not in assigned]
    rows=[];report=dict(model=a.model,model_sha256=weight_hash,shapes_sha256=sha256(a.shapes),
                        chains=chains,rows=rows,complete=False,note=__doc__)
    for index,(name,m) in enumerate(modules.items()):
        parts=[torch.randn(shape,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last) for shape in records[name]['parts']]
        nxt=modules[successors[name]] if name in successors else None
        residual=parts[-1] if name in residuals else None;s=m.spec
        m.joint_warps=0;m.joint_next_mode=0
        expected=m(ChannelConcat(parts),residual=residual)
        prepared=None
        if name in predecessors:
            prepared=op.prepare_cached_cat(parts,m.quant_multiplier,m.activation,s['bits'],s['padded_channels'],m.sparse_channel_map,m.sparse_channels,5,True)
        modes=list(MODES) if nxt is not None else ['separate','joint']
        row=dict(name=name,parts=records[name]['parts'],successor=successors.get(name),residual=residual is not None,modes=modes,timing={},bitwise_equal={})
        for kind in (['full','prefix'] if prepared is not None else ['full']):
            row['timing'][kind]={};row['bitwise_equal'][kind]={}
            for mode in (modes if index%2==0 else list(reversed(modes))):
                warps,next_mode,produces=MODES[mode]
                m.joint_warps=warps;m.joint_next_mode=next_mode
                if nxt is not None:nxt.joint_warps=8
                distinct=produces and nxt is not None and nxt.shared_skip_planar and nxt.shared_skip_prefix_planar and nxt.uses_shared_pointwise()
                costs={};checks={}
                for following in ((0,8) if nxt is not None else (None,)):
                    key='none' if following is None else str(following)
                    if following==8 and not distinct:
                        costs[key],checks[key]=costs['0'],checks['0'];continue
                    if nxt is not None:nxt.joint_warps=following
                    def call():
                        prefix=(m,*prepared,parts[0].shape[1]) if kind=='prefix' else None
                        return m(ChannelConcat(parts,prepared=prefix),next_quantizer=nxt if produces else None,residual=residual)
                    actual=call();y=actual[0] if isinstance(actual,(tuple,list)) else actual
                    equal=torch.equal(y,expected)
                    if not equal:raise AssertionError((name,kind,mode,following,float((y-expected).abs().max())))
                    if nxt is not None and produces:
                        next_parts=[y,*parts];needed=not(distinct and following==8)
                        full=op.prepare_cached_cat(next_parts,nxt.quant_multiplier,nxt.activation,nxt.spec['bits'],nxt.spec['padded_channels'],nxt.sparse_channel_map,nxt.sparse_channels,5,True)
                        filled=op.prepare_next_cat(next_parts,nxt.quant_multiplier,nxt.activation,nxt.spec['bits'],nxt.spec['padded_channels'],nxt.sparse_channel_map,nxt.sparse_channels,actual[1],actual[2],s['shape'][0],5,needed)
                        if not torch.equal(full[0],filled[0]) or (needed and not torch.equal(full[1],filled[1])):
                            raise AssertionError((name,kind,mode,following,'prefix'))
                        if not needed:assert filled[1].untyped_storage().nbytes()==1
                        del full,filled,next_parts
                    checks[key]=True;del actual,y
                    costs[key]=timing(call,warmup=2,repeats=a.repeats,graph=True,graph_batch=5)
                row['timing'][kind][mode]=costs;row['bitwise_equal'][kind][mode]=checks
        rows.append(row);report.update(count=len(rows),complete=len(rows)==157,environment=metadata());write_json(a.output,report)
        print(index,name,{kind:{mode:{key:round(t['median_ms'],5) for key,t in values.items()} for mode,values in costs.items()} for kind,costs in row['timing'].items()},flush=True)
        del expected,prepared,parts,residual
    keyed={row['name']:row for row in rows};chosen={};scores=[]
    for chain in chains:
        selected,score=solve_chain(chain,keyed,a.relative_margin,a.absolute_margin_ms,a.prefix_margin_ms);chosen.update(selected);scores.append(score)
    plan=dict(format='quantdemoire-joint-plan-v1',model_sha256=weight_hash,source_report=a.output,source_report_sha256=sha256(a.output),
              measured_environment=report['environment'],layers={name:MODES[mode][0] for name,mode in chosen.items()},
              joint_next_modes={name:MODES[mode][1] for name,mode in chosen.items()},selection='chain_joint_producer_dynamic_programming',
              relative_margin=a.relative_margin,absolute_margin_ms=a.absolute_margin_ms,prefix_margin_ms=a.prefix_margin_ms,
              penalized_cost_ms=sum(scores),note='Enable joint-next globally to use per-layer modes. Whole-network timing and validation are required; allocator and cross-layer cache effects are not predicted.')
    write_json(a.plan,plan);print(dict(Counter(chosen.values())),flush=True)


if __name__=='__main__':main()
