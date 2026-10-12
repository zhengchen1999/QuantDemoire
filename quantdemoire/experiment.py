import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .deploy import load_model, sha256
from .esdnet import ESDNet


def model_for(backend, path, device="cuda"):
    if backend in ("fake", "cuda", "fake_fp16"):
        return load_model(path, backend, device)
    m = ESDNet(48, 32, 64, 32, 1)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return m.eval().to(device=device, dtype=torch.float16 if backend == "fp16" else torch.float32,
                       memory_format=torch.channels_last)


def pad_image(x):
    h, w = x.shape[-2:]
    dh, dw = (-h) % 32, (-w) % 32
    t, l = dh // 2, dw // 2
    if dh or dw:
        x = torch.cat([F.pad(x[:, c:c+1], (l, dw-l, t, dh-t), value=mean)
                       for c, mean in enumerate((.3827, .4141, .3912))], 1)
    return x, (t, l, h, w)


@torch.inference_mode()
def timing(fn, warmup=5, repeats=20, graph=False, graph_batch=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(graph_batch):
                static_output = fn()
        fn = g.replay
    values = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / (graph_batch if graph else 1))
    return dict(mode="cuda_graph" if graph else "eager", graph_batch=graph_batch if graph else 1, median_ms=float(np.median(values)), p90_ms=float(np.percentile(values, 90)),
                min_ms=min(values), samples_ms=values)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


@torch.inference_mode()
def shapes(height=224, width=224):
    m = ESDNet(48, 32, 64, 32, 1).eval().cuda().half().to(memory_format=torch.channels_last)
    records = []
    handles = []
    for name, module in m.named_modules():
        if isinstance(module, nn.Conv2d):
            def hook(layer, inputs, output, name=name):
                records.append(dict(name=name, weight=list(layer.weight.shape),
                                    input=list(inputs[0].shape), output=list(output.shape),
                                    stride=layer.stride[0], padding=layer.padding[0], dilation=layer.dilation[0]))
            handles.append(module.register_forward_hook(hook))
    x, _ = pad_image(torch.zeros(1, 3, height, width, device="cuda", dtype=torch.float16))
    m(x.contiguous(memory_format=torch.channels_last))
    for h in handles:
        h.remove()
    return records


def read_json(path):
    return json.loads(Path(path).read_text())


def run_module(module, *arguments, environment=None, log=None):
    """Use a fresh process with explicit execution settings for each experiment."""
    import os
    import subprocess
    import sys
    env = {key: value for key, value in os.environ.items() if not key.startswith('QUANTDEMOIRE_')}
    env.update(environment or {})
    command = [sys.executable, '-m', module, *map(str, arguments)]
    if log is None:
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True, env=env)
    else:
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as stream:
            try:
                subprocess.run(command, check=True, env=env, stdout=stream, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f'Command failed; see {log}') from error
