import argparse
import json
from pathlib import Path
from quantdemoire.deploy import export_legacy
import torch

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--quant-state", required=True)
    p.add_argument("--bits", type=int, choices=[4, 8], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tail-fraction", type=float)
    p.add_argument("--weight-range", choices=['stored','minmax','mse'], default='stored')
    p.add_argument('--main-output',choices=['float32','float16'],default='float32',
                   help='corrected integer-branch intermediate; accumulation remains INT32/FP32')
    a = p.parse_args()
    torch.set_num_threads(4)
    c = export_legacy(a.checkpoint, a.quant_state, a.bits, a.output, a.tail_fraction, a.weight_range)
    for layer in c['layers'].values():
        if layer['kind']=='integer':layer['main_output']=a.main_output
    (Path(a.output)/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    print(f"Exported {len(c['layers'])} layers to {a.output}")
