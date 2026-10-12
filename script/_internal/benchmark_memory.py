"""Compare resident and peak GPU memory against FP32 and FP16 references."""
import argparse
from pathlib import Path

from script._internal.common import (benchmark_arguments, benchmark_preflight,
                                    modes, read_json, run, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    benchmark_arguments(parser)
    parser.add_argument("--output", default="results/reproduction/memory")
    args = parser.parse_args()
    benchmark_preflight(args)
    output = Path(args.output).resolve()
    summary = {}
    for mode in modes(args.mode):
        report = output / f"{mode}.json"
        run("measure_memory.py", "--checkpoint", args.checkpoint,
            "--w4", args.w4, "--w8", args.w8, "--image", args.image,
            "--runtime-profile", "--include-fp32", "--output", report,
            *(["--graph"] if mode == "graph" else []))
        rows = read_json(report)
        keys = ("model_allocated_bytes", "peak_allocated_bytes", "peak_reserved_bytes",
                "model_memory_saving_vs_fp16", "peak_memory_saving_vs_fp16",
                "model_memory_saving_vs_fp32", "peak_memory_saving_vs_fp32")
        summary[mode] = {name: {key: row[key] for key in keys if key in row}
                         for name, row in rows.items()}
        print(f"\n{mode}: model / peak allocated (GiB) / saving vs FP16")
        for name, row in rows.items():
            saving = f"{100 * row['peak_memory_saving_vs_fp16']:.2f}%" if name.startswith("w") else "—"
            print(f"{name}: {row['peak_allocated_bytes'] / 2**30:.3f} GiB / {saving}")
    write_json(output / "memory.json", summary)


if __name__ == "__main__":
    main()
