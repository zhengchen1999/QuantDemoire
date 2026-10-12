"""JIT-build the CUDA backend. No dense floating-point fallback is hidden here."""
import os
import hashlib
import json
import importlib.util
from pathlib import Path
from functools import lru_cache

import torch

_build_failure = None
loaded_source_sha256 = None


@lru_cache(None)
def extension():
    global _build_failure, loaded_source_sha256
    if _build_failure is not None:
        raise RuntimeError('CUDA extension build failed earlier in this process; fix the build and restart') from _build_failure
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major != 8:
        raise RuntimeError("This release supports SM80/86/89; other GPUs need validation")
    from torch.utils import cpp_extension
    cuda = os.environ.get("CUDA_HOME", cpp_extension.CUDA_HOME)
    cutlass = Path(os.environ.get("CUTLASS_ROOT", ".third_party/cutlass")).resolve()
    if not cuda or not (Path(cuda) / "bin/nvcc").is_file():
        raise RuntimeError("Set CUDA_HOME to a CUDA 12.1 toolkit (including nvcc)")
    if not (cutlass / "include/cutlass/cutlass.h").is_file():
        raise RuntimeError("Run uv run python -m kernel or set CUTLASS_ROOT to CUTLASS 3.5.1")
    cpp_extension.CUDA_HOME = cuda
    import ninja
    os.environ["PATH"] = str(ninja.BIN_DIR) + os.pathsep + os.environ["PATH"]
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    os.environ.setdefault("MAX_JOBS", "2")
    sources=[Path(__file__).parent/'csrc'/f for f in ("bindings.cpp", "conv.cu", "epilogue.cu", "fused.cu", "sparse_tc.cu", "half_branch.cu", "pointwise.cu", "sparse_tuned.cu", "joint.cu", "joint_two.cu", "prepare_second.cu", "joint_omit.cu", "joint_omit_two.cu", "sparse_omit.cu")]
    dependencies=sources+[Path(__file__).parent/'csrc'/'joint_common.cuh']
    digest=hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in dependencies)).hexdigest()
    fingerprint=dict(source_sha256=digest,torch=torch.__version__,cuda=str(Path(cuda).resolve()),
                     cutlass=str(cutlass),arch=os.environ['TORCH_CUDA_ARCH_LIST'],
                     compiler=os.environ.get('CXX','c++'),flags='O3,c++17,lineinfo,relaxed-constexpr')
    # Some synchronized workspaces refresh source mtimes without changing the
    # contents. A content-checked successful build avoids repeated recompiles.
    directory=Path(cpp_extension._get_build_directory('quantdemoire_integer_v1',False))
    manifest=directory/'verified_build.json'
    if manifest.is_file():
        saved=json.loads(manifest.read_text())
        binary=Path(saved.get('binary',''))
        if saved.get('fingerprint')==fingerprint and binary.is_file() and hashlib.sha256(binary.read_bytes()).hexdigest()==saved.get('binary_sha256'):
            spec=importlib.util.spec_from_file_location(saved['module'],binary)
            result=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(result)
            loaded_source_sha256=digest
            return result
    try:
        result=cpp_extension.load(
        name="quantdemoire_integer_v1",
        sources=[str(p) for p in sources],
        extra_include_paths=[str(cutlass / "include")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "--expt-relaxed-constexpr"],
        verbose=os.environ.get("QUANTDEMOIRE_BUILD_VERBOSE") == "1",
        )
    except Exception as error:
        _build_failure=error
        raise
    loaded_source_sha256=digest
    binary=Path(result.__file__)
    payload=dict(fingerprint=fingerprint,binary=str(binary),module=result.__name__,
                 binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    temporary=manifest.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,indent=2)+'\n')
    temporary.replace(manifest)
    return result
