"""Full-network latency and GPU memory reproduction."""
import os
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from quantdemoire.precision import configure, metadata
from quantdemoire.experiment import model_for, pad_image, timing, write_json
from quantdemoire.deploy import sha256
from quantdemoire.runtime import CudaGraphRunner


@torch.inference_mode()
def measure_main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["fp32", "fp16", "fake", "fake_fp16", "cuda"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--height", type=int, default=2160)
    p.add_argument("--width", type=int, default=3840)
    p.add_argument("--image", help="Use this RGB image at its native resolution instead of random input; image loading and padding are outside timing")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--output", required=True)
    p.add_argument("--graph", action="store_true")
    p.add_argument('--memory-format',choices=('channels_last','contiguous'),default='channels_last')
    p.add_argument('--runtime-profile',action='store_true',help='Apply the CUDA model folder runtime.json before loading; explicit environment overrides are preserved')
    p.add_argument('--changing-input',action='store_true',help='graph runner including a GPU input copy on every inference')
    a = p.parse_args()
    if a.warmup < 0 or a.repeats < 1:
        p.error("warmup must be nonnegative and repeats must be positive")
    if a.memory_format=='contiguous' and a.backend not in ('fp32','fake'):
        p.error('contiguous benchmarking is supported for the FP32 reference paths')
    if a.runtime_profile:
        if a.backend != 'cuda':p.error('--runtime-profile requires the CUDA backend')
        from quantdemoire.profiles import apply_execution_profile
        apply_execution_profile(a.model)
    configure()
    torch.manual_seed(123)
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    model = model_for(a.backend, a.model)
    layout=torch.channels_last if a.memory_format=='channels_last' else torch.contiguous_format
    model.to(memory_format=layout)
    model_bytes = torch.cuda.memory_allocated() - before
    dtype = torch.float32 if a.backend in ('fp32','fake') else torch.float16
    if a.image:
        with Image.open(a.image) as image:
            x = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32) / 255).permute(2,0,1)[None].cuda()
        original_input_hw = list(x.shape[-2:])
        x, padding = pad_image(x)
        x = x.to(dtype)
    else:
        original_input_hw = [a.height, a.width]
        x, padding = pad_image(torch.rand(1,3,a.height,a.width,device="cuda",dtype=dtype))
    x = x.contiguous(memory_format=layout)
    out = model(x)
    if not all(torch.isfinite(y).all() for y in out):
        raise RuntimeError("nonfinite model output")
    del out
    torch.cuda.synchronize()
    # Include lazy dequantized caches in fake backends' resident model storage.
    model_bytes = torch.cuda.memory_allocated() - before - x.untyped_storage().nbytes()
    # Warm first so autotuner workspace is excluded from steady-state peaks.
    for _ in range(a.warmup):
        model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if a.changing_input:
        if a.backend in ('fp32','fake'):raise ValueError('changing-input graph runner uses FP16 features')
        runner=CudaGraphRunner(model,x)
        result=timing(lambda:runner(x),warmup=2,repeats=a.repeats)
        result['mode']='cuda_graph_with_input_copy'
    else:
        result = timing(lambda: model(x), warmup=0, repeats=a.repeats, graph=a.graph)
    result.update(backend=a.backend, model=a.model, input=list(x.shape), environment=metadata(),
                  cudnn_autotune=torch.backends.cudnn.benchmark,
                  memory_format=a.memory_format,
                  image=str(Path(a.image).resolve()) if a.image else None,
                  input_sha256=sha256(a.image) if a.image else None,
                  original_input_hw=original_input_hw, valid_region_top_left_height_width=list(padding),
                  warmup=a.warmup, repeats=a.repeats, benchmark_script_sha256=sha256(__file__),
                  runtime_profile_sha256=sha256(Path(a.model)/'runtime.json') if a.runtime_profile else None,
                  model_sha256=sha256(Path(a.model)/'model.safetensors' if a.backend in ('cuda','fake','fake_fp16') else a.model),
                  config_sha256=sha256(Path(a.model)/'config.json') if (Path(a.model)/'config.json').is_file() else None,
                  model_allocated_bytes=model_bytes, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), includes="complete network, all three outputs, quantization, packing, sparse branch; excludes image I/O, input transfer/padding and metrics")
    write_json(a.output, result)
    print(result)


def benchmark_main(kind):
    import statistics
    from quantdemoire.experiment import read_json, run_module
    from quantdemoire.profiles import profile_environment

    parser = argparse.ArgumentParser(description=f"Compare complete-network {kind} in independent GPU processes.")
    parser.add_argument("--checkpoint", default="checkpoints/uhdm_checkpoint.pth")
    parser.add_argument("--w4", default="checkpoints/reproduction/deploy/w4a4")
    parser.add_argument("--w8", default="checkpoints/reproduction/deploy/w8a8")
    parser.add_argument("--image", default="data/UHDM-full/test/0000_moire.jpg")
    parser.add_argument("--mode", choices=("graph", "eager", "both"), default="both")
    parser.add_argument("--output", default=f"outputs/reproduction/{kind}")
    if kind == "speed":
        parser.add_argument("--rounds", type=int, default=3)
        parser.add_argument("--repeats", type=int, default=30)
        parser.add_argument("--warmup", type=int, default=7)
    else:
        parser.set_defaults(rounds=1, repeats=10, warmup=5)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("rounds/repeats must be positive; warmup must be nonnegative")
    for name in ("checkpoint", "image"):
        path = Path(getattr(args, name)).resolve()
        if not path.is_file():
            parser.error(f"Missing {name}: {path}")
        setattr(args, name, str(path))
    profiles = []
    for name in ("w4", "w8"):
        path = Path(getattr(args, name)).resolve()
        profiles.append(profile_environment(path))
        setattr(args, name, str(path))
    keys = ("QUANTDEMOIRE_FUSE_ATTENTION", "QUANTDEMOIRE_FUSE_RESIDUAL", "QUANTDEMOIRE_RELEASE_FEATURES")
    baseline = {key: "1" if any(profile.get(key) == "1" for profile in profiles) else "0" for key in keys}
    jobs = [("fp16", "fp16", args.checkpoint), ("w4a4", "cuda", args.w4), ("w8a8", "cuda", args.w8)]
    if kind == "memory":
        jobs.insert(0, ("fp32", "fp32", args.checkpoint))
    output = Path(args.output).resolve()
    summary = {}
    # Clear the aggregate completion marker before starting a new measurement.
    write_json(output / f"{kind}.json", dict(complete=False))
    for mode in (("graph", "eager") if args.mode == "both" else (args.mode,)):
        report = dict(complete=False, mode=mode, image=args.image,
                      rounds=args.rounds, warmup=args.warmup, repeats=args.repeats, rows=[])
        write_json(output / f"{mode}.json", report)
        measured = {}
        for iteration in range(args.rounds):
            for name, backend, model in (jobs if iteration % 2 == 0 else list(reversed(jobs))):
                raw = output / f"{mode}_{name}_r{iteration}.json"
                options = ["--runtime-profile"] if backend == "cuda" else []
                if backend == "fp32":
                    options += ["--memory-format", "contiguous"]
                if mode == "graph":
                    options += ["--graph"]
                run_module("quantdemoire.benchmark", "measure", "--backend", backend,
                           "--model", model, "--image", args.image, "--warmup", args.warmup,
                           "--repeats", args.repeats, "--output", raw, *options,
                           environment=baseline if backend in ("fp32", "fp16") else None,
                           log=raw.with_suffix(".log"))
                measured[name] = read_json(raw)
                report["rows"].append(dict(round=iteration, variant=name,
                    median_ms=measured[name]["median_ms"],
                    peak_allocated_bytes=measured[name]["peak_allocated_bytes"], raw=str(raw)))
                write_json(output / f"{mode}.json", report)
                print(f"{mode}: {name}, round {iteration + 1} complete", flush=True)
        if kind == "speed":
            rows = {name: dict(
                median_ms=statistics.median(row["median_ms"] for row in report["rows"] if row["variant"] == name),
                peak_allocated_bytes=max(row["peak_allocated_bytes"] for row in report["rows"] if row["variant"] == name))
                for name, _, _ in jobs}
            print(f"\n{mode}: model / latency (ms) / speedup vs FP16")
            for name, row in rows.items():
                row["speedup_vs_fp16"] = rows["fp16"]["median_ms"] / row["median_ms"]
                print(f"{name}: {row['median_ms']:.3f} ms / {row['speedup_vs_fp16']:.3f}x")
        else:
            keys = ("model_allocated_bytes", "peak_allocated_bytes", "peak_reserved_bytes")
            rows = {name: {key: value[key] for key in keys} for name, value in measured.items()}
            for name in ("w4a4", "w8a8"):
                for reference in ("fp16", "fp32"):
                    for label, key in (("model", "model_allocated_bytes"), ("peak", "peak_allocated_bytes")):
                        rows[name][f"{label}_memory_saving_vs_{reference}"] = 1 - rows[name][key] / rows[reference][key]
            print(f"\n{mode}: model / peak allocated (GiB) / saving vs FP16")
            for name, row in rows.items():
                saving = f"{100 * row['peak_memory_saving_vs_fp16']:.2f}%" if name.startswith("w") else "—"
                print(f"{name}: {row['peak_allocated_bytes'] / 2**30:.3f} GiB / {saving}")
        report.update(complete=True, summary=rows)
        write_json(output / f"{mode}.json", report)
        summary[mode] = rows
    write_json(output / f"{kind}.json", dict(complete=True, modes=summary))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in ("measure", "speed", "memory"):
        raise SystemExit("Usage: python -m quantdemoire.benchmark {speed,memory,measure} [options]")
    kind = sys.argv.pop(1)
    if kind == "measure":
        measure_main()
    else:
        benchmark_main(kind)
