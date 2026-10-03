"""Check dense-counterfactual arithmetic and full-model fusion coverage."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED'] = '1'
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from kernel.diagnostic import extension
from quantdemoire.branch_diagnostic import main_only
from quantdemoire.deploy import PackedConv, ChannelConcat, load_model, sha256
from quantdemoire.experiment import pad_image, write_json
from quantdemoire.precision import configure, metadata
from quantdemoire.profiles import apply_execution_profile
from quantdemoire.runtime import CudaGraphRunner


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    configure(); apply_execution_profile(args.model, overwrite=True)
    os.environ['QUANTDEMOIRE_RELEASE_FEATURES'] = '1'
    model = load_model(args.model, 'cuda'); extension()
    with Image.open(args.image) as image:
        x = torch.from_numpy(np.asarray(image.convert('RGB'), dtype=np.float32)/255).permute(2,0,1)[None].cuda()
    x, _ = pad_image(x); x = x.half().contiguous(memory_format=torch.channels_last)
    counts = Counter()
    def observe(module, inputs, kwargs):
        counts['layers'] += 1
        for key in ('next_quantizer','future_quantizer','residual','omit_feature'):
            counts[key] += int(kwargs.get(key) is not None and kwargs.get(key) is not False)
        source = inputs[0]
        counts['prepared'] += int(isinstance(source, ChannelConcat) and source.prepared is not None)
        counts['second_cached'] += int(isinstance(source, ChannelConcat) and source.second_cached is not None)
    modules = [m for m in model.modules() if isinstance(m, PackedConv)]
    hooks = [m.register_forward_pre_hook(observe, with_kwargs=True) for m in modules]
    model(x); full_counts = dict(counts); counts.clear()
    with main_only(model):
        actual = model(x); dense_counts = dict(counts)
    for hook in hooks:
        hook.remove()
    assert full_counts == dense_counts, (full_counts, dense_counts)
    for m in modules:
        m.sparse_enabled = False
    expected = model(x)
    for m in modules:
        m.sparse_enabled = True
    equal = [torch.equal(a, b) for a, b in zip(actual, expected)]
    assert all(equal), [(a-b).abs().max().item() for a,b in zip(actual, expected)]
    with main_only(model):
        runner = CudaGraphRunner(model, x)
        captured = runner(x)
    graph_equal = [torch.equal(a,b) for a,b in zip(captured, expected)]
    assert all(graph_equal)
    report = dict(complete=True, accuracy_valid=False, input=list(x.shape),
        image_sha256=sha256(args.image), weights_sha256=sha256(Path(args.model)/'model.safetensors'),
        full_fusion_counts=full_counts, dense_fusion_counts=dense_counts,
        dense_reference_bitwise_equal=equal, dense_graph_bitwise_equal=graph_equal,
        environment=metadata())
    write_json(args.output, report)
    print(report)


if __name__ == '__main__':
    main()
