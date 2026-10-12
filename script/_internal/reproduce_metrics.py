"""Calibrate/export W4A4 and W8A8, then evaluate full UHDM quality metrics."""
import argparse
from pathlib import Path

from script._internal.common import read_json, run, write_json
from script._internal.runtime import prepare_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "calibrate", "export", "evaluate"), default="all")
    parser.add_argument("--recipe", choices=("deploy", "paper"), default="deploy",
                        help="adjusted PTQ: 0.25%% or 0.5%% tails per side")
    parser.add_argument("--bits", type=int, choices=(4, 8), nargs="+", default=[4, 8])
    parser.add_argument("--checkpoint", default="checkpoints/uhdm_checkpoint.pth")
    parser.add_argument("--calibration", default="data/UHDM-paper-calibration/train")
    parser.add_argument("--data", default="data/UHDM-full/test")
    parser.add_argument("--artifacts", help="calibration state directory")
    parser.add_argument("--models", help="output root containing w4a4/ and w8a8/")
    parser.add_argument("--output", help="directory for metrics and validation JSON")
    parser.add_argument("--limit", type=int, help="development subset; omit for full 500-pair results")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    for name, default in (("artifacts", f"outputs/reproduction/{args.recipe}"),
                          ("models", f"checkpoints/reproduction/{args.recipe}"),
                          ("output", f"results/reproduction/{args.recipe}")):
        setattr(args, name, Path(getattr(args, name) or default).resolve())
    if not Path(args.checkpoint).is_file():
        parser.error(f"missing checkpoint: {args.checkpoint}")
    if args.stage in ("all", "evaluate"):
        pairs = sorted(Path(args.data).rglob("*_gt.jpg"))
        if not pairs or (args.limit is None and len(pairs) != 500):
            parser.error("full evaluation requires 500 test pairs; --limit is for development only")
        if any(not path.with_name(path.name.replace("_gt.jpg", "_moire.jpg")).is_file() for path in pairs):
            parser.error("test split contains an unpaired ground-truth image")
    bits_list = list(dict.fromkeys(args.bits))
    for bits in bits_list:
        model = args.models / f"w{bits}a{bits}"
        stages = ("calibrate", "export") if args.stage == "all" else (args.stage,)
        for stage in stages:
            if stage not in ("calibrate", "export"):
                continue
            run("reproduce_ptq.py", "--stage", stage, "--recipe", args.recipe,
                "--bits", bits, "--small-tensor-policy", "full_if_pooled",
                "--sampling-rate", 0.01 if bits == 4 else 0.05,
                "--checkpoint", args.checkpoint, "--calibration", args.calibration,
                "--test", args.data, "--artifacts", args.artifacts,
                "--model-directory", model, "--results", args.output / "packing")
        if args.stage in ("all", "export", "evaluate"):
            prepare_runtime(model, bits)
    if args.stage not in ("all", "evaluate"):
        return
    subset = ["--limit", args.limit] if args.limit else []
    results = {}
    for backend in ("fp32", "fp16"):
        report = args.output / f"{backend}.json"
        layout = "contiguous" if backend == "fp32" else "channels_last"
        run("evaluate.py", "--backend", backend, "--model", args.checkpoint,
            "--data", args.data, "--memory-format", layout, "--autotune",
            "--resume", "--output", report, *subset,
            environment={"QUANTDEMOIRE_FUSE_ATTENTION": "1",
                         "QUANTDEMOIRE_FUSE_RESIDUAL": "1",
                         "QUANTDEMOIRE_RELEASE_FEATURES": "1"})
        results[backend] = read_json(report)
    for bits in bits_list:
        model = args.models / f"w{bits}a{bits}"
        run("verify_graph_fusions.py", "--backend", "cuda", "--model", model,
            "--data", args.data, "--runtime-profile", "--compare-feature-release",
            "--graph", "--autotune", "--output", args.output / f"validation_w{bits}.json", *subset)
        for backend in ("fake", "cuda"):
            name = f"w{bits}a{bits}_{backend}"
            report = args.output / f"{name}.json"
            options = ["--runtime-profile"] if backend == "cuda" else ["--memory-format", "contiguous"]
            run("evaluate.py", "--backend", backend, "--model", model,
                "--data", args.data, "--autotune", "--resume", "--output", report,
                *options, *subset)
            results[name] = read_json(report)
    summary = {name: {"count": result["count"], "complete": result["complete"],
                      **result["mean"]} for name, result in results.items()}
    write_json(args.output / "metrics.json", dict(recipe=args.recipe,
               full_test=all(row["count"] == 500 for row in summary.values()), models=summary))
    print("Model                  PSNR       SSIM       LPIPS")
    for name, row in summary.items():
        print(f"{name:22} {row['psnr']:.6f}  {row['ssim']:.6f}  {row['lpips']:.6f}")


if __name__ == "__main__":
    main()
