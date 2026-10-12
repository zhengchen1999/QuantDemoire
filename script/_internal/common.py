"""Shared paths and process isolation for the three reproduction commands."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
INTERNAL = Path(__file__).resolve().parent
DEFAULT_MODELS = "checkpoints/reproduction/deploy"


def clean_environment():
    return {key: value for key, value in os.environ.items()
            if not key.startswith("QUANTDEMOIRE_")}


def run(script, *arguments, environment=None):
    command = [sys.executable, str(INTERNAL / script), *map(str, arguments)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True,
                   env={**clean_environment(), **(environment or {})})


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def benchmark_arguments(parser):
    parser.add_argument("--checkpoint", default="checkpoints/uhdm_checkpoint.pth")
    parser.add_argument("--w4", default=f"{DEFAULT_MODELS}/w4a4")
    parser.add_argument("--w8", default=f"{DEFAULT_MODELS}/w8a8")
    parser.add_argument("--image", default="data/UHDM-full/test/0000_moire.jpg")
    parser.add_argument("--mode", choices=("graph", "eager", "both"), default="both")


def benchmark_preflight(args):
    from quantdemoire.profiles import profile_environment
    for name in ("checkpoint", "image"):
        path = Path(getattr(args, name)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
        setattr(args, name, str(path))
    profiles = []
    for name in ("w4", "w8"):
        path = Path(getattr(args, name)).resolve()
        profiles.append(profile_environment(path))
        setattr(args, name, str(path))
    keys = ("QUANTDEMOIRE_FUSE_ATTENTION", "QUANTDEMOIRE_FUSE_RESIDUAL",
            "QUANTDEMOIRE_RELEASE_FEATURES")
    return {key: "1" if any(profile.get(key) == "1" for profile in profiles) else "0"
            for key in keys}


def modes(mode):
    return ("graph", "eager") if mode == "both" else (mode,)
