"""Measure each deployment in a fresh process, including sparse metadata/workspace."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from quantdemoire.experiment import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--w4',required=True)
    p.add_argument('--w8',required=True)
    p.add_argument('--height',type=int,default=2160)
    p.add_argument('--width',type=int,default=3840)
    p.add_argument('--image',help='use the same real input for every baseline and deployment')
    p.add_argument('--graph',action='store_true')
    p.add_argument('--include-fp32',action='store_true',help='also measure a strict FP32 NCHW baseline in a fresh process')
    p.add_argument('--runtime-profile',action='store_true',help='Use each model folder execution profile and enable applicable fusions for the FP16 baseline')
    p.add_argument('--output',default='results/memory.json')
    a=p.parse_args();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    baseline_env=dict(os.environ)
    if a.runtime_profile:
        from quantdemoire.profiles import profile_environment
        profiles=[profile_environment(folder) for folder in (a.w4,a.w8)]
        for key in ('QUANTDEMOIRE_FUSE_ATTENTION','QUANTDEMOIRE_FUSE_RESIDUAL','QUANTDEMOIRE_RELEASE_FEATURES'):
            baseline_env.setdefault(key,'1' if any(profile.get(key)=='1' for profile in profiles) else '0')
    results={}
    jobs=[('fp16','fp16',a.checkpoint),('w4a4','cuda',a.w4),('w8a8','cuda',a.w8)]
    if a.include_fp32:jobs.insert(0,('fp32','fp32',a.checkpoint))
    for name,backend,model in jobs:
        target=out.with_name(out.stem+'_'+name+'.json')
        command=[sys.executable,str(Path(__file__).with_name('benchmark_e2e.py')),'--backend',backend,'--model',model,
                 '--height',str(a.height),'--width',str(a.width),'--repeats','10','--output',str(target)]
        if a.graph:command.append('--graph')
        if a.image:command.extend(['--image',a.image])
        if backend=='fp32':command+=['--memory-format','contiguous']
        if a.runtime_profile and backend=='cuda':command.append('--runtime-profile')
        subprocess.run(command,check=True,env=baseline_env if backend in ('fp16','fp32') else None)
        results[name]=json.loads(target.read_text())
    for name in ('w4a4','w8a8'):
        results[name]['model_memory_saving_vs_fp16']=1-results[name]['model_allocated_bytes']/results['fp16']['model_allocated_bytes']
        results[name]['peak_memory_saving_vs_fp16']=1-results[name]['peak_allocated_bytes']/results['fp16']['peak_allocated_bytes']
        if a.include_fp32:
            results[name]['model_memory_saving_vs_fp32']=1-results[name]['model_allocated_bytes']/results['fp32']['model_allocated_bytes']
            results[name]['peak_memory_saving_vs_fp32']=1-results[name]['peak_allocated_bytes']/results['fp32']['peak_allocated_bytes']
    write_json(out,results)


if __name__=='__main__':main()
