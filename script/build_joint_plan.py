"""Create a local, weight-bound dispatch proposal from a complete shape sweep."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--output',required=True)
    p.add_argument('--relative-margin',type=float,default=.03)
    p.add_argument('--absolute-margin-ms',type=float,default=.002)
    p.add_argument('--pointwise-only',action='store_true')
    a=p.parse_args();source=Path(a.report);report=json.loads(source.read_text())
    if not report['complete'] or report['layer_count']!=len(report['rows']):raise ValueError('incomplete shape sweep')
    if a.relative_margin<0 or a.absolute_margin_ms<0:raise ValueError('margins must be nonnegative')
    layers={}
    for row in report['rows']:
        if not all(row['bitwise_equal'].values()):raise ValueError('shape sweep contains unequal outputs')
        t={name:v['median_ms'] for name,v in row['timings'].items()};name=min(('joint4','joint8'),key=lambda n:t[n])
        gain=t['separate']-t[name]
        choose=gain>max(a.absolute_margin_ms,t['separate']*a.relative_margin)
        if a.pointwise_only and row['shape']['weight'][2]!=1:choose=False
        layers[row['layer']]=int(name[-1]) if choose else 0
    out=dict(format='quantdemoire-joint-plan-v1',model_sha256=report['model_sha256'],
             source_report=str(source),source_report_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             measured_environment=report['environment'],relative_margin=a.relative_margin,
             absolute_margin_ms=a.absolute_margin_ms,pointwise_only=a.pointwise_only,layers=layers,
             note='Offline proposal: validate whole-network latency before deployment. Choices are correct at other input sizes, but tuned for the recorded native UHDM shapes. Preparation and producer-prefix interactions are excluded from the source microbenchmark.')
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(out,indent=2)+'\n')
    print(dict(Counter(layers.values())))


if __name__=='__main__':main()
