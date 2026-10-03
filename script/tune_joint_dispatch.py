"""Tune layer dispatch including prepare and producer-prefix interactions.

Each DenseBlock is a chain: a separate producer can write its consumer's
quantized prefix, while a joint producer currently cannot. Measure both input
states and the next consumer mode, because a shared-input consumer can omit
its producer's planar prefix. Solve these adjacent dependencies by dynamic programming. Whole-network validation
and timing are still required before adopting the resulting plan.
"""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED'] = '1'
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import torch
from kernel.ops import extension
from quantdemoire.deploy import load_model, PackedConv, ChannelConcat, sha256
from quantdemoire.esdnet import DenseBlock, ResidualDenseBlock
from quantdemoire.experiment import timing, write_json
from quantdemoire.precision import configure, metadata


def solve_chain(chain, rows, relative_margin, absolute_margin):
    def cost(name, kind, mode, following):
        entry = rows[name]['timing'][kind][str(mode)]
        if 'median_ms' in entry:  # Earlier reports had no consumer-dependent cost.
            return entry['median_ms']
        return entry['none' if following is None else str(following)]['median_ms']

    @lru_cache(None)
    def visit(index, previous, current):
        kind = 'prefix' if previous == 0 else 'full'
        following_modes = (0, 8) if index + 1 < len(chain) else (None,)
        candidates = []
        for following in following_modes:
            value = cost(chain[index], kind, current, following)
            if current:
                value += max(absolute_margin, cost(chain[index], kind, 0, following) * relative_margin)
            tail_score, tail = visit(index + 1, current, following) if following is not None else (0., ())
            candidates.append((value + tail_score, (current, *tail)))
        return min(candidates)

    score, modes = min(visit(0, None, mode) for mode in (0, 8))
    return dict(zip(chain, modes)), score


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--shapes', required=True, help='Complete benchmark_prepare report for this model')
    p.add_argument('--output', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--repeats', type=int, default=8)
    p.add_argument('--relative-margin', type=float, default=.01)
    p.add_argument('--absolute-margin-ms', type=float, default=.001)
    a = p.parse_args()
    if int(os.environ.get('QUANTDEMOIRE_JOINT_NEXT_QUANT','0')):
        p.error('use script/tune_joint_next.py when joint producers are enabled')
    if a.repeats < 1 or min(a.relative_margin, a.absolute_margin_ms) < 0:
        p.error('invalid timing or selection settings')
    configure(); torch.manual_seed(993)
    source = json.loads(Path(a.shapes).read_text())
    weight_hash = sha256(Path(a.model) / 'model.safetensors')
    if not source['complete'] or source['count'] != 157 or source['weights_sha256'] != weight_hash:
        raise ValueError('shape report does not cover this model')
    model = load_model(a.model, 'cuda'); op = extension()
    modules = {name: m for name, m in model.named_modules() if isinstance(m, PackedConv)}
    records = {row['name']: row for row in source['rows']}
    if set(modules) != set(records):
        raise ValueError('shape report and model layer names differ')
    chains, successors, predecessors, residuals = [], {}, {}, set()
    assigned = set()
    for name, block in model.named_modules():
        if not isinstance(block, DenseBlock):
            continue
        chain = [f'{name}.conv_layers.{i}.conv.0' for i in range(len(block.conv_layers))]
        chain.append(f'{name}.conv_post.conv')
        for left, right in zip(chain, chain[1:]):
            modules[left].joint_warps = 0
            if not modules[left].can_quantize_for(modules[right]):
                raise ValueError(f'unsupported chain edge: {left} -> {right}')
            successors[left] = right; predecessors[right] = left
        if isinstance(block, ResidualDenseBlock):
            residuals.add(chain[-1])
        chains.append(chain); assigned.update(chain)
    chains += [[name] for name in modules if name not in assigned]
    rows = []
    report = dict(model=a.model, model_sha256=weight_hash, shapes_sha256=sha256(a.shapes),
                  chains=chains, rows=rows, complete=False, note=__doc__)
    for index, (name, m) in enumerate(modules.items()):
        parts = [torch.randn(shape, device='cuda', dtype=torch.float16).contiguous(memory_format=torch.channels_last)
                 for shape in records[name]['parts']]
        s = m.spec
        nxt = modules[successors[name]] if name in successors else None
        residual = parts[-1] if name in residuals else None
        m.joint_warps = 0
        expected = m(ChannelConcat(parts), residual=residual)
        prepared = None
        if name in predecessors:
            prepared = op.prepare_cached_cat(parts, m.quant_multiplier, m.activation, s['bits'],
                s['padded_channels'], m.sparse_channel_map, m.sparse_channels,
                {'pair': 2, 'quad': 3, 'quad_loop': 4, 'quad_swizzle': 5}[m.prepare_impl], True)
            if parts[0].shape[1] != modules[predecessors[name]].spec['shape'][0]:
                raise ValueError('prefix shape mismatch')
        row = dict(name=name, parts=records[name]['parts'], successor=successors.get(name),
                   residual=residual is not None, timing={}, bitwise_equal={})
        kinds = ['full', 'prefix'] if prepared is not None else ['full']
        for kind in kinds:
            row['timing'][kind] = {}; row['bitwise_equal'][kind] = {}
            for mode in ((0, 8) if index % 2 == 0 else (8, 0)):
                m.joint_warps = mode
                costs, checks = {}, {}
                next_modes = (0, 8) if nxt is not None else (None,)
                if nxt is not None:
                    nxt.joint_warps = 8
                different_output = (mode == 0 and nxt is not None and nxt.shared_skip_planar
                                    and nxt.shared_skip_prefix_planar and nxt.uses_shared_pointwise())
                for following in next_modes:
                    key = 'none' if following is None else str(following)
                    if following == 8 and not different_output:
                        costs[key], checks[key] = costs['0'], checks['0']
                        continue
                    if nxt is not None:
                        nxt.joint_warps = following
                    def call():
                        prefix = (m, *prepared, parts[0].shape[1]) if kind == 'prefix' else None
                        return m(ChannelConcat(parts, prepared=prefix),
                                 next_quantizer=nxt if mode == 0 else None, residual=residual)
                    actual = call()
                    y = actual[0] if isinstance(actual, (tuple, list)) else actual
                    equal = torch.equal(y, expected)
                    checks[key] = equal
                    if not equal:
                        raise AssertionError((name, kind, mode, following, float((y - expected).abs().max())))
                    if different_output and following == 8:
                        assert actual[2].untyped_storage().nbytes() == 1
                    del actual, y
                    costs[key] = timing(call, warmup=2, repeats=a.repeats, graph=True, graph_batch=5)
                row['timing'][kind][str(mode)] = costs
                row['bitwise_equal'][kind][str(mode)] = checks
        rows.append(row)
        report.update(count=len(rows), complete=len(rows) == 157, environment=metadata())
        write_json(a.output, report)
        print(index, name, {kind: {mode: {n: round(t['median_ms'], 5) for n, t in by_next.items()} for mode, by_next in costs.items()}
                            for kind, costs in row['timing'].items()}, flush=True)
        del expected, prepared, parts, residual
    keyed = {row['name']: row for row in rows}; layers = {}; scores = []
    for chain in chains:
        chosen, score = solve_chain(chain, keyed, a.relative_margin, a.absolute_margin_ms)
        layers.update(chosen); scores.append(score)
    plan = dict(format='quantdemoire-joint-plan-v1', model_sha256=weight_hash,
                source_report=a.output, source_report_sha256=sha256(a.output),
                measured_environment=report['environment'], layers=layers,
                relative_margin=a.relative_margin, absolute_margin_ms=a.absolute_margin_ms,
                selection='chain_dynamic_programming_with_consumer_cost', penalized_cost_ms=sum(scores),
                note='Includes per-layer preparation, CSR merge and producer-prefix costs. Requires whole-network timing and validation; does not predict allocator or cross-layer cache effects.')
    write_json(a.plan, plan)
    print(dict(Counter(layers.values())), flush=True)


if __name__ == '__main__':
    main()
