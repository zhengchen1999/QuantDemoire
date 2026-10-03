"""Measure FP16 branch cost on real ESDNet activations and the saved schedule.

The dense counterfactual is timing-only and is never evaluated for image quality.
Each producer owns its output packing; each consumer owns the remaining input
preparation. No preparation operation is assigned to two layers.
"""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED'] = '1'
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from kernel import ops
from kernel import diagnostic
from quantdemoire.branch_diagnostic import dense_core, main_only
from quantdemoire.deploy import ChannelConcat, PackedConv, load_model, sha256
from quantdemoire.experiment import pad_image, timing, write_json
from quantdemoire.precision import configure, metadata
from quantdemoire.profiles import apply_execution_profile


def clone_input(x):
    # Cache containers are consumed once; immutable tensor storage can be shared.
    return ChannelConcat(x.parts, x.prepared, x.second_cached) if isinstance(x, ChannelConcat) else x


class CoreRecorder:
    def __init__(self, backend):
        self.backend = backend
        self.calls = []

    def __getattr__(self, name):
        fn = getattr(self.backend, name)
        if name != 'fused_conv' and not name.startswith('joint_'):
            return fn
        def record(*args, **kwargs):
            result = fn(*args, **kwargs)
            self.calls.append((name, args))
            return result
        return record


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--graph-batch', type=int, default=3)
    parser.add_argument('--max-layers', type=int, default=0, help='diagnostic smoke test only')
    parser.add_argument('--release-features', type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    if min(args.repeats, args.rounds, args.graph_batch) < 1 or args.warmup < 1 or args.max_layers < 0:
        parser.error('positive measurement counts are required')
    configure(); torch.manual_seed(509)
    environment = apply_execution_profile(args.model, overwrite=True)
    os.environ['QUANTDEMOIRE_RELEASE_FEATURES'] = str(args.release_features)
    environment['QUANTDEMOIRE_RELEASE_FEATURES'] = str(args.release_features)
    model = load_model(args.model, 'cuda')
    backend = ops.extension(); diagnostic_backend = diagnostic.extension()
    with Image.open(args.image) as image:
        x = torch.from_numpy(np.asarray(image.convert('RGB'), dtype=np.float32)/255).permute(2, 0, 1)[None].cuda()
    x, _ = pad_image(x)
    x = x.half().contiguous(memory_format=torch.channels_last)
    expected_count = sum(isinstance(m, PackedConv) for m in model.modules())
    report = dict(complete=False, accuracy_valid=False, input=list(x.shape),
        image_sha256=sha256(args.image), model_sha256=sha256(Path(args.model)/'model.safetensors'),
        config_sha256=sha256(Path(args.model)/'config.json'), runtime=environment,
        production_kernel_sha256=ops.loaded_source_sha256,
        diagnostic_binary_sha256=sha256(diagnostic_backend.__file__),
        source_sha256={path:sha256(path) for path in ('kernel/diagnostic.py',
            'quantdemoire/branch_diagnostic.py', 'quantdemoire/deploy.py', __file__)},
        definitions=dict(core='matched integer convolution and output epilogue, no input preparation or outgoing packing',
            pipeline='input preparation + integer convolution + outgoing caches/residual; sparse work disabled',
            mixed='same layer and real full-model inputs, including sparse work and all owned preparation',
            R_core='sum(mixed - pipeline) / sum(core)',
            R_pipeline='sum(mixed - pipeline) / sum(pipeline)'),
        caveats=['Counterfactual outputs intentionally omit tail weights and are invalid for quality.',
                 'Four-warp separate producers fold the remaining dense prefix into the same integer tile.',
                 'Core timing writes the ordinary FP16 convolution output even for an omitted-feature producer.',
                 'Layer sums are isolated CUDA Graph measurements, not full-network latency.'],
        expected_layers=expected_count, rows=[], network=[])
    write_json(args.output, report)
    # Complete-network ablation is reported separately, never used to enlarge
    # the integer-core denominator with floating-point network operations.
    for round_index in range(args.rounds):
        for mode in (('mixed', 'main_only') if round_index % 2 == 0 else ('main_only', 'mixed')):
            with main_only(model) if mode == 'main_only' else nullcontext():
                result = timing(lambda:model(x), warmup=args.warmup, repeats=args.repeats, graph=True)
            report['network'].append(dict(round=round_index, path=mode, timing=result))
    write_json(args.output, report)
    saved_forwards = []

    def install(name, m):
        forward = m.forward
        saved_forwards.append((m, forward))
        def measure(source, *positional, **kwargs):
            if positional:
                raise RuntimeError('unexpected positional fused arguments')
            if args.max_layers and len(report['rows']) >= args.max_layers:
                return forward(source, **kwargs)
            frozen = clone_input(source)
            recorder = CoreRecorder(backend)
            with patch('kernel.ops.extension', return_value=recorder):
                output = forward(source, **kwargs)
            if len(recorder.calls) != 1:
                raise RuntimeError(f'{name}: expected one integer convolution, got {len(recorder.calls)}')
            operation, values = recorder.calls[0]
            packed, planar = values[:2]
            locations = m.compact_locations if m.compact_sparse else m.locations
            row = dict(layer=name, input=list(frozen.shape), bits=m.spec['bits'],
                shape=m.spec['shape'], joint_warps=m.joint_warps,
                integer_entrypoint=operation, nnz=m.values.numel(),
                prefixed_input=isinstance(frozen, ChannelConcat) and frozen.prepared is not None,
                second_cached_input=isinstance(frozen, ChannelConcat) and frozen.second_cached is not None,
                next_quantizer=kwargs.get('next_quantizer') is not None,
                future_quantizer=kwargs.get('future_quantizer') is not None,
                residual=kwargs.get('residual') is not None, omit_feature=kwargs.get('omit_feature', False),
                measurements=[])
            def full_path():
                return forward(clone_input(frozen), **kwargs)
            def core():
                return dense_core(m, packed, planar, locations, kwargs.get('residual'))
            for round_index in range(args.rounds):
                order = ('mixed','core','main_only') if (round_index+len(report['rows'])) % 2 == 0 else ('main_only','core','mixed')
                for mode in order:
                    with main_only(m) if mode == 'main_only' else nullcontext():
                        result = timing(core if mode == 'core' else full_path,
                            warmup=args.warmup, repeats=args.repeats, graph=True, graph_batch=args.graph_batch)
                    row['measurements'].append(dict(round=round_index, path=mode, timing=result))
            medians = {mode:statistics.median(r['timing']['median_ms'] for r in row['measurements'] if r['path']==mode)
                       for mode in ('mixed','core','main_only')}
            delta = medians['mixed']-medians['main_only']
            row.update(median_ms=medians, incremental_ms=delta,
                       R_core=delta/medians['core'], R_pipeline=delta/medians['main_only'])
            report['rows'].append(row)
            write_json(args.output, report)
            print(f"{len(report['rows']):3d} {name}: +{delta:.4f} ms, R_core={row['R_core']:.1%}", flush=True)
            return output
        m.forward = measure

    for name, m in model.named_modules():
        if isinstance(m, PackedConv):
            install(name, m)
    try:
        model(x)
    finally:
        for m, forward in saved_forwards:
            m.forward = forward
    rows = report['rows']
    totals = {mode:sum(r['median_ms'][mode] for r in rows) for mode in ('mixed','core','main_only')}
    delta = totals['mixed']-totals['main_only']
    report.update(complete=len(rows)==expected_count, measured_layers=len(rows),
        totals_ms=totals, incremental_ms=delta, R_core=delta/totals['core'],
        R_pipeline=delta/totals['main_only'],
        layers_below_30pct=sum(r['R_core']<=.3 for r in rows),
        aggregate_below_30pct=(delta/totals['core']<=.3) if len(rows)==expected_count else None,
        environment=metadata(), hotspots=[r['layer'] for r in sorted(rows,key=lambda r:-r['incremental_ms'])])
    write_json(args.output, report)
    print(json.dumps({k:v for k,v in report.items() if k in ('complete','totals_ms','R_core','R_pipeline','layers_below_30pct')}, indent=2))


if __name__ == '__main__':
    main()
