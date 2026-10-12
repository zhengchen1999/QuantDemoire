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
def main():
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


if __name__ == "__main__":
    main()
