"""Reproduce ESDNet/UHDM calibration, exported weights and quality metrics."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import yaml

from quantdemoire.deploy import sha256
from quantdemoire.experiment import read_json, run_module, write_json
from quantdemoire.export import export_model, prepare_runtime, verify_export

ROOT = Path.cwd()


def validate_calibration_source(run, checkpoint, expected):
    """Do not label or combine a saved PTQ state with a different source recipe."""
    run=Path(run)
    provenance=json.loads((run/'provenance.json').read_text())
    if provenance['config_sha256']!=sha256(run/'config.yaml'):
        raise ValueError('saved calibration config was modified')
    if provenance['checkpoint_sha256']!=sha256(checkpoint):
        raise ValueError('export checkpoint differs from calibration checkpoint')
    saved=yaml.safe_load((run/'config.yaml').read_text())
    keys=('N_BITS_W','N_BITS_A','ACT_SAMPLE_RATE','INPUT_CHANNEL_SAMPLE_RATE',
          'ACT_SMALL_TENSOR_POLICY','CALIBRATION_AGGREGATION','SMOOTH_ALPHA',
          'WEIGHT_OUTLIER_PERCENTILE','WEIGHT_OUTLIER_MODE')
    for key in keys:
        if saved['QUANTIZATION'].get(key)!=expected['QUANTIZATION'].get(key):
            raise ValueError(f'export recipe differs from calibration: {key}')


def configuration(recipe,bits,policy,checkpoint,calibration,test,run,aggregation='mean',sampling_rate=.001,smooth_alpha=.5):
    config=yaml.safe_load((ROOT/f'config/ptq/{recipe}_w{bits}a{bits}.yaml').read_text())
    config['DATA'].update(CALI_DATASET=str(Path(calibration).resolve()),
        TRAIN_DATASET=str(Path(calibration).resolve()),TEST_DATASET=str(Path(test).resolve()))
    config['TEST']['LOAD_PATH']=str(Path(checkpoint).resolve())
    config['GENERAL'].update(SAVE_PREFIX=str(run.parent),EXP_NAME=run.name)
    config['QUANTIZATION'].update(ACT_SMALL_TENSOR_POLICY=policy,CALIBRATION_AGGREGATION=aggregation,
                                 QUANT_STATE_PATH=str(run/'quant_state.pth'),
                                 ACT_SAMPLE_RATE=sampling_rate,INPUT_CHANNEL_SAMPLE_RATE=sampling_rate)
    config['QUANTIZATION']['SMOOTH_ALPHA']=smooth_alpha
    return config



def prepare_model(args, bits, model):
    sampling_rate = 0.01 if bits == 4 else 0.05
    label = f"{args.recipe}-w{bits}a{bits}-full_if_pooled-g{sampling_rate:g}"
    run = args.artifacts / label
    config = configuration(args.recipe, bits, "full_if_pooled", args.checkpoint,
                           args.calibration, args.data, run, sampling_rate=sampling_rate)
    if args.stage in ("calibrate", "all"):
        if (run / "quant_state.pth").exists():
            raise FileExistsError(f"Calibration already exists: {run}; use --stage export/evaluate or a new --artifacts directory")
        run.mkdir(parents=True, exist_ok=True)
        config_path = run / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        with tarfile.open(run / "source.tar.gz", "w:gz") as snapshot:
            for path in sorted(Path(__file__).parent.rglob("*.py")):
                snapshot.add(path, arcname=str(Path("quantdemoire") / path.relative_to(Path(__file__).parent)))
            for name in ("uv.lock", "pyproject.toml"):
                snapshot.add(ROOT / name, arcname=name)
        run_module("quantdemoire.calibrate", "--config", config_path, "--calibrate-only")
    if args.stage in ("export", "all"):
        state = run / "quant_state.pth"
        if not state.is_file():
            raise FileNotFoundError(f"Missing calibrated state: {state}")
        validate_calibration_source(run, args.checkpoint, config)
        if (model / "runtime.json").exists():
            raise FileExistsError("Output already has an execution profile; use a new --models directory to preserve its provenance")
        export_model(args.checkpoint, state, bits, model)
        for name in ("provenance.json", "calibration_history.json", "config.yaml"):
            shutil.copyfile(run / name, model / name)
        write_json(model / "reproduction.json", dict(
            recipe=args.recipe, bits=bits, small_tensor_policy="full_if_pooled",
            quant_state_sha256=sha256(state), calibration_aggregation="mean",
            gamma1=sampling_rate, gamma2=sampling_rate, smooth_alpha=0.5,
            source_archive_sha256=sha256(run / "source.tar.gz"),
            mask_policy="preserved from calibration", published_quantized_result_reproduced=False))
        verify_export(model, args.checkpoint, state, args.output / "packing" / label / "format.json")


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
                          ("output", f"outputs/reproduction/{args.recipe}/metrics")):
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
        prepare_model(args, bits, model)
        if args.stage in ("all", "export", "evaluate"):
            prepare_runtime(model, bits)
    if args.stage not in ("all", "evaluate"):
        return
    subset = ["--limit", args.limit] if args.limit else []
    results = {}
    for backend in ("fp32", "fp16"):
        report = args.output / f"{backend}.json"
        layout = "contiguous" if backend == "fp32" else "channels_last"
        run_module("quantdemoire.evaluate", "--backend", backend, "--model", args.checkpoint,
            "--data", args.data, "--memory-format", layout, "--autotune",
            "--resume", "--output", report, *subset,
            environment={"QUANTDEMOIRE_FUSE_ATTENTION": "1",
                         "QUANTDEMOIRE_FUSE_RESIDUAL": "1",
                         "QUANTDEMOIRE_RELEASE_FEATURES": "1"})
        results[backend] = read_json(report)
    for bits in bits_list:
        model = args.models / f"w{bits}a{bits}"
        run_module("quantdemoire.evaluate", "verify", "--backend", "cuda", "--model", model,
            "--data", args.data, "--runtime-profile", "--compare-feature-release",
            "--graph", "--autotune", "--output", args.output / f"validation_w{bits}.json", *subset)
        for backend in ("fake", "cuda"):
            name = f"w{bits}a{bits}_{backend}"
            report = args.output / f"{name}.json"
            options = ["--runtime-profile"] if backend == "cuda" else ["--memory-format", "contiguous"]
            run_module("quantdemoire.evaluate", "--backend", backend, "--model", model,
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
