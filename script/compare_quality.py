"""Compare the same hashed image subset; never mix paper and local absolute PSNR."""
import argparse
import json
from pathlib import Path
import numpy as np
from quantdemoire.experiment import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fp32',required=True)
    p.add_argument('--results',nargs='+',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    baseline=json.loads(Path(a.fp32).read_text())
    if baseline['backend']!='fp32' or not baseline.get('complete'):raise ValueError('complete FP32 reference required')
    ref={r['image']:r for r in baseline['rows']}
    output=dict(reference=a.fp32,count=len(ref),paper_psnr_gap=dict(w8a8=.12,w4a4=1.04),results=[])
    for path in a.results:
        result=json.loads(Path(path).read_text());rows={r['image']:r for r in result['rows']}
        if not result.get('complete') or rows.keys()!=ref.keys() or result['crop']!=baseline['crop'] or result['protocol']!=baseline['protocol']:
            raise ValueError(f'incompatible evaluation protocol/subset: {path}')
        for name,r in rows.items():
            if (r['gt_sha256'],r['input_sha256'])!=(ref[name]['gt_sha256'],ref[name]['input_sha256']):
                raise ValueError(f'image hash mismatch: {name}')
        output['results'].append(dict(file=path,backend=result['backend'],model_sha256=result['model_sha256'],mean=result['mean'],
            psnr_gap=float(np.mean([ref[n]['psnr']-r['psnr'] for n,r in rows.items()])),
            ssim_gap=float(np.mean([ref[n]['ssim']-r['ssim'] for n,r in rows.items()])),
            lpips_increase=float(np.mean([r['lpips']-ref[n]['lpips'] for n,r in rows.items()]))))
    write_json(a.output,output)
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
