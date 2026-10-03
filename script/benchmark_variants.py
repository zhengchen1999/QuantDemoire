"""Run an explicit variant matrix sequentially, with isolated GPU processes."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--repeats',type=int,default=30)
    p.add_argument('--warmup',type=int,default=7)
    p.add_argument('--image',help='Use the same real image for every variant')
    p.add_argument('--height',type=int,default=2160)
    p.add_argument('--width',type=int,default=3840)
    p.add_argument('--graph',action='store_true')
    a=p.parse_args()
    if a.rounds<1 or a.repeats<1:p.error('rounds and repeats must be positive')
    if a.warmup<0:p.error('warmup must be nonnegative')
    config=json.loads(Path(a.config).read_text());variants=config['variants']
    if len({v['name'] for v in variants})!=len(variants):raise ValueError('duplicate variant name')
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    result=dict(config=config,rounds=a.rounds,repeats=a.repeats,warmup=a.warmup,image=a.image,graph=a.graph,rows=[],complete=False)
    for iteration in range(a.rounds):
        for v in (variants if iteration%2==0 else list(reversed(variants))):
            if not v['name'].replace('_','').replace('-','').isalnum():raise ValueError('invalid variant name')
            raw=target.with_name(f'{target.stem}_{v["name"]}_r{iteration}.json')
            env={**os.environ,**config.get('environment',{}),**v.get('environment',{})}
            command=[sys.executable,str(Path(__file__).with_name('benchmark_e2e.py')),'--backend',v['backend'],'--model',v['model'],'--output',str(raw),'--warmup',str(a.warmup),'--repeats',str(a.repeats),'--height',str(a.height),'--width',str(a.width)]
            if a.image:command.extend(['--image',a.image])
            if v.get('memory_format'):command.extend(['--memory-format',v['memory_format']])
            if a.graph:command.append('--graph')
            with raw.with_suffix('.log').open('w') as log:
                subprocess.run(command,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
            measured=json.loads(raw.read_text())
            row=dict(round=iteration,variant=v['name'],backend=v['backend'],median_ms=measured['median_ms'],peak_allocated_bytes=measured['peak_allocated_bytes'],raw=str(raw))
            result['rows'].append(row)
            target.write_text(json.dumps(result,indent=2)+'\n');print(row,flush=True)
    result['summary']={v['name']:dict(median_ms=statistics.median(r['median_ms'] for r in result['rows'] if r['variant']==v['name']),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in result['rows'] if r['variant']==v['name'])) for v in variants}
    result['complete']=True
    target.write_text(json.dumps(result,indent=2)+'\n');print(result['summary'],flush=True)


if __name__=='__main__':main()
