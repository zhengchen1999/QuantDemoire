"""Prepare pinned CUTLASS sources and compile the integer CUDA extension."""
import argparse
import os
from pathlib import Path
import subprocess

CUTLASS_COMMIT = "f7b19de32c5d1f3cedfc735c2849f12b537522ee"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    cuda = Path(os.environ.get("CUDA_HOME", ""))
    if not (cuda / "bin/nvcc").is_file():
        parser.error("set CUDA_HOME to a CUDA 12.1 toolkit containing bin/nvcc")
    cutlass = Path(os.environ.get("CUTLASS_ROOT", ".third_party/cutlass")).resolve()
    if not cutlass.exists():
        cutlass.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", "v3.5.1",
                        "https://github.com/NVIDIA/cutlass.git", str(cutlass)], check=True)
    commit = subprocess.check_output(["git", "-C", str(cutlass), "rev-parse", "HEAD"], text=True).strip()
    if commit != CUTLASS_COMMIT or not (cutlass / "include/cutlass/cutlass.h").is_file():
        parser.error(f"CUTLASS_ROOT must contain CUTLASS v3.5.1 ({CUTLASS_COMMIT})")
    os.environ["CUTLASS_ROOT"] = str(cutlass)
    from kernel.ops import extension
    print(extension().__file__)


if __name__ == "__main__":
    main()
