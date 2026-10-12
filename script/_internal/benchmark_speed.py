"""Compare complete-network FP16/W4A4/W8A8 latency in alternating processes."""
import argparse
from pathlib import Path

from script._internal.common import (benchmark_arguments, benchmark_preflight,
                                    modes, read_json, run, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    benchmark_arguments(parser)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=7)
    parser.add_argument("--output", default="results/reproduction/speed")
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("rounds/repeats must be positive; warmup must be nonnegative")
    baseline = benchmark_preflight(args)
    output = Path(args.output).resolve()
    config = output / "variants.json"
    write_json(config, dict(variants=[
        dict(name="fp16", backend="fp16", model=args.checkpoint, environment=baseline),
        dict(name="w4a4", backend="cuda", model=args.w4, runtime_profile=True),
        dict(name="w8a8", backend="cuda", model=args.w8, runtime_profile=True),
    ]))
    summary = {}
    for mode in modes(args.mode):
        report = output / f"{mode}.json"
        run("benchmark_variants.py", "--config", config, "--image", args.image,
            "--rounds", args.rounds, "--warmup", args.warmup, "--repeats", args.repeats,
            "--output", report, *(["--graph"] if mode == "graph" else []))
        rows = read_json(report)["summary"]
        for row in rows.values():
            row["speedup_vs_fp16"] = rows["fp16"]["median_ms"] / row["median_ms"]
        summary[mode] = rows
        print(f"\n{mode}: model / latency (ms) / speedup vs FP16")
        for name, row in rows.items():
            print(f"{name}: {row['median_ms']:.3f} ms / {row['speedup_vs_fp16']:.3f}x")
    write_json(output / "speed.json", summary)


if __name__ == "__main__":
    main()
