"""Alternating, isolated-process end-to-end ablation of producer quantization."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--output',required=True,help='summary JSON; raw runs are saved next to it')
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--repeats',type=int,default=30)
    p.add_argument('--height',type=int,default=2160)
    p.add_argument('--width',type=int,default=3840)
    p.add_argument('--graph',action='store_true')
    a=p.parse_args()
    if a.rounds<1 or a.repeats<1:p.error('rounds and repeats must be positive')
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for iteration in range(a.rounds):
        for enabled in ((False,True) if iteration%2==0 else (True,False)):
            output=target.with_name(f'{target.stem}_{"on" if enabled else "off"}_r{iteration}.json')
            env=dict(os.environ,QUANTDEMOIRE_FUSE_NEXT_QUANT=str(int(enabled)),QUANTDEMOIRE_PREPARE_IMPL='pair',QUANTDEMOIRE_COMPACT_SPARSE='1',QUANTDEMOIRE_MAIN_HALF='1',QUANTDEMOIRE_SPARSE_IMPL='csr')
            command=[sys.executable,str(Path(__file__).with_name('benchmark_e2e.py')),'--backend','cuda','--model',a.model,'--output',str(output),'--warmup','7','--repeats',str(a.repeats),'--height',str(a.height),'--width',str(a.width)]
            if a.graph:command.append('--graph')
            subprocess.run(command,env=env,check=True,stdout=subprocess.DEVNULL)
            record=json.loads(output.read_text())
            rows.append(dict(round=iteration,enabled=enabled,median_ms=record['median_ms'],peak_allocated_bytes=record['peak_allocated_bytes'],raw=str(output)))
            print(rows[-1],flush=True)
    off=statistics.median(row['median_ms'] for row in rows if not row['enabled'])
    on=statistics.median(row['median_ms'] for row in rows if row['enabled'])
    summary=dict(model=a.model,graph=a.graph,rows=rows,off_median_ms=off,on_median_ms=on,speedup=off/on,latency_reduction_percent=100*(1-on/off),note='Median of isolated-run medians; alternating order, same weights, pair preparation in both modes. Image I/O excluded.')
    target.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
