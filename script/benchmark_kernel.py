import os
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tests"))

import torch
from helpers import synthetic_layer
from kernel.ops import extension
from quantdemoire.precision import configure, metadata
from quantdemoire.experiment import shapes, timing, write_json
from quantdemoire.deploy import load_model, PackedConv


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--height",type=int,default=224)
    p.add_argument("--width",type=int,default=224)
    p.add_argument("--repeats",type=int,default=30)
    p.add_argument("--output",default="results/kernel.json")
    p.add_argument("--correctness-only",action="store_true")
    p.add_argument("--model", help="use every deployed layer's actual sparse weights")
    p.add_argument("--bits", type=int, choices=[4,8], nargs="+", default=[4,8])
    p.add_argument("--tail-fraction", type=float, default=.0025)
    a=p.parse_args()
    configure();torch.manual_seed(123)
    records=shapes(a.height,a.width)
    write_json(Path(a.output).with_name("esdnet_shapes.json"),records)
    # Cover every executed layer shape; dedup only identical runtime signatures.
    unique={tuple(r["weight"]+r["input"]+[r["stride"],r["padding"],r["dilation"]]):r for r in records}
    deployed=load_model(a.model,'cuda',fuse_relu=False) if a.model else None
    selected=records if deployed else list(unique.values())
    op=extension();rows=[]
    for i,r in enumerate(selected):
        for bits in a.bits:
            actual_module=deployed.get_submodule(r['name']) if deployed else None
            if isinstance(actual_module,PackedConv):
                m=actual_module
                if m.spec['bits']!=bits:raise ValueError('model bit width differs')
            else:
                m=synthetic_layer(r,bits,tail_fraction=a.tail_fraction).cuda()
            x=torch.randn(r["input"],device="cuda",dtype=torch.float16).contiguous(memory_format=torch.channels_last)
            actual=m(x)
            expected=m.fake_forward(x)
            error=(actual.float()-expected).abs()
            entry=dict(layer=r["name"],shape=r,bits=bits,weight_source="deployment" if isinstance(actual_module,PackedConv) else "synthetic",max_abs=error.max().item(),
                       rms=error.square().mean().sqrt().item(),ref_rms=expected.square().mean().sqrt().item())
            entry["numerical_pass"]=entry["rms"]<max(1e-4,entry["ref_rms"]*.003)
            if not a.correctness_only:
                s=m.spec;n,c,h,w=x.shape;k,_,kr,_=s["shape"]
                packed,planar=op.prepare(x,m.inverse_smooth,m.activation,bits,s["padded_channels"])
                def main_conv():
                    if m.main_half:
                        return op.fused_conv(packed,planar,m.packed_weight,m.output_scale,m.row,m.locations,m.values,
                                             m.epilogue_sums,m.bias,m.activation,bits,s['padded_channels'],k,kr,
                                             s['stride'],s['padding'],s['dilation'],False,False)
                    return op.integer_conv(packed,m.packed_weight,m.output_scale,bits,n,h,w,s["padded_channels"],k,kr,s["stride"],s["padding"],s["dilation"])
                entry["main"]=timing(main_conv,repeats=a.repeats,graph=True,graph_batch=50)
                entry["mixed"]=timing(lambda:m(x),repeats=a.repeats,graph=True,graph_batch=50)
                m.sparse_enabled=False
                entry["without_sparse"]=timing(lambda:m(x),repeats=a.repeats,graph=True,graph_batch=50)
                m.sparse_enabled=True
                entry["incremental_branch_ratio"]=(entry["mixed"]["median_ms"]-entry["without_sparse"]["median_ms"])/entry["main"]["median_ms"]
                entry['branch_ratio_vs_main_pipeline']=(entry['mixed']['median_ms']-entry['without_sparse']['median_ms'])/entry['without_sparse']['median_ms']
                entry["branch_30pct_pass"]=entry["incremental_branch_ratio"]<=.30
                wh=m.dense_weight().half().contiguous(memory_format=torch.channels_last);bh=m.bias.half()
                entry["fp16"]=timing(lambda:torch.nn.functional.conv2d(x,wh,bh,s["stride"],s["padding"],s["dilation"]),repeats=a.repeats,graph=True,graph_batch=50)
            rows.append(entry)
            write_json(a.output,dict(environment=metadata(),network_conv_count=len(records),unique_runtime_shapes=len(unique),rows=rows))
            print(i,bits,r["weight"],r["input"],entry["max_abs"],entry["rms"],entry.get("incremental_branch_ratio"),flush=True)
    if not all(r["numerical_pass"] for r in rows):
        raise SystemExit("Numerical checks failed; inspect output JSON")


if __name__=="__main__":main()
