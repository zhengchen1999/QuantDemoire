"""Explicit, non-exportable main-only counterfactual for CUDA timing."""
from contextlib import contextmanager

from .deploy import PackedConv


@contextmanager
def main_only(model):
    modules = [m for m in model.modules() if isinstance(m, PackedConv)]
    for m in modules:
        if not (m.backend == 'cuda' and m.main_half and m.sparse_impl == 'csr'
                and m.compact_sparse and m.prepare_impl == 'quad_swizzle'
                and m.joint_warps in (0, 8)):
            raise ValueError('matched branch diagnostic requires the deployed quad_swizzle CSR profile')
    saved = [(m.sparse_enabled, m.diagnostic_main_only) for m in modules]
    try:
        for m in modules:
            m.sparse_enabled = False
            m.diagnostic_main_only = True
        yield
    finally:
        for m, (sparse, diagnostic) in zip(modules, saved):
            m.sparse_enabled = sparse
            m.diagnostic_main_only = diagnostic


def _joint_args(m, packed, planar, locations, residual, *, packed_metadata=False):
    s = m.spec
    k, _, r, _ = s['shape']
    h, w = planar.shape[-2:]
    balanced = m.balance_csr and m.joint_warps == 8 and m.joint_variant in (8, 9)
    return (packed, planar, m.packed_weight, m.output_scale,
        m.balanced_row if balanced else m.row,
        m.locations if m.uses_shared_pointwise() else locations,
        m.sparse_linear_offsets(locations, h, w, s['dilation'], packed=packed_metadata),
        m.values, m.epilogue_sums, m.bias, m.activation,
        residual if residual is not None else m.values,
        s['bits'], s['padded_channels'], k, r, s['stride'], s['padding'], s['dilation'],
        s.get('relu', False), False, residual is not None, m.joint_warps or 4, m.joint_variant), balanced


def dense_core(m, packed, planar, locations, residual=None):
    """Integer convolution + its output epilogue, excluding consumer packing."""
    from kernel.ops import extension
    op = extension()
    if m.joint_warps or residual is not None:
        args, balanced = _joint_args(m, packed, planar, locations, residual)
        return op.joint_conv(*args, balanced, m.retain_shared, m.few_channels, False)
    s = m.spec
    k, _, r, _ = s['shape']
    return op.fused_conv(packed, planar, m.packed_weight, m.output_scale,
        m.row, locations, m.values, m.epilogue_sums, m.bias, m.activation,
        s['bits'], s['padded_channels'], k, r, s['stride'], s['padding'], s['dilation'],
        s.get('relu', False), False)


def dense_forward(m, packed, planar, locations, next_quantizer=None,
                  residual=None, future_quantizer=None, omit_feature=False):
    if m.sparse_enabled or not m.diagnostic_main_only:
        raise ValueError('dense counterfactual must be explicitly enabled and tails disabled')
    if next_quantizer is None:
        return dense_core(m, packed, planar, locations, residual)
    from kernel.diagnostic import extension
    op = extension()
    name = 'dense_next' + ('_two' if future_quantizer is not None else '')
    name += '_omitted' if omit_feature else ''
    args, balanced = _joint_args(m, packed, planar, locations, residual)
    nxt = next_quantizer
    extra = () if future_quantizer is None else (
        future_quantizer.quant_multiplier, future_quantizer.activation,
        future_quantizer.spec['bits'], 32)
    result = getattr(op, name)(*args, nxt.quant_multiplier, nxt.activation,
        nxt.sparse_channel_map, nxt.spec['bits'], nxt.spec['padded_channels'],
        nxt.sparse_channels, False, 0, balanced, m.retain_shared, m.few_channels, False, *extra)
    if omit_feature:
        result[0]._quantdemoire_omitted = True
    return result
