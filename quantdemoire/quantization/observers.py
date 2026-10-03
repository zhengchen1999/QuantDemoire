"""Quantization range observers."""

import math

import torch

MAX_DIRECT_PERCENTILE_NUMEL = 2_000_000


def ste_round(tensor):
    return (torch.round(tensor) - tensor).detach() + tensor


def rank_tail_mask(tensor, fraction):
    """Retain both tails by stable rank, including enough tied extrema."""
    if not 0 < fraction < .5 or tensor.numel() == 0:
        raise ValueError('tail fraction must be between zero and one half')
    count = math.ceil(tensor.numel()*fraction)
    order = tensor.detach().flatten().argsort(stable=True)
    mask = torch.zeros(tensor.numel(), dtype=torch.bool, device=tensor.device)
    mask[order[:count]] = True
    mask[order[-count:]] = True
    return mask.reshape_as(tensor)


def _as_tensor(value, reference):
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.tensor(value, device=reference.device, dtype=reference.dtype)


def resolve_sample_size(numel, sample_size=None, sample_rate=None):
    if sample_rate is not None and sample_rate > 0:
        return max(1, min(numel, math.ceil(numel * sample_rate)))
    return sample_size


def sample_tensor_values(tensor, sample_size=None, sample_rate=None):
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        return flat
    sample_size = resolve_sample_size(flat.numel(), sample_size, sample_rate)
    if sample_size is None or sample_size <= 0 or flat.numel() <= sample_size:
        return flat
    indices = torch.randint(0, flat.numel(), (sample_size,), device=flat.device)
    return flat[indices]


def estimate_channel_absmax(tensor, sample_size=None, sample_rate=None):
    if tensor.ndim != 4:
        raise ValueError("estimate_channel_absmax expects an NCHW tensor.")

    samples = tensor.detach().permute(1, 0, 2, 3).reshape(tensor.size(1), -1)
    sample_size = resolve_sample_size(samples.size(1), sample_size, sample_rate)
    if sample_size is not None and sample_size > 0 and samples.size(1) > sample_size:
        indices = torch.randint(0, samples.size(1), (sample_size,), device=samples.device)
        samples = samples[:, indices]
    return samples.abs().amax(dim=1).clamp_min_(1e-6)


def _clamp_percentile_bounds(lower_percentile, upper_percentile):
    lower = max(0.0, min(float(lower_percentile), 100.0))
    upper = max(lower, min(float(upper_percentile), 100.0))
    return lower, upper


def estimate_range(
    tensor,
    method="minmax",
    sample_size=None,
    sample_rate=None,
    lower_percentile=0.1,
    upper_percentile=99.9,
    num_bits=4,
    small_tensor_policy="legacy",
):
    if small_tensor_policy not in {"legacy", "full_if_single", "full_if_pooled"}:
        raise ValueError(f"Unknown small tensor sampling policy: {small_tensor_policy}")
    if tensor.numel() == 0:
        zero = torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
        return zero, zero

    method = method.lower()
    use_sampling = method in {
        "sample_minmax",
        "sample_percentile",
        "quantdemoire_act",
        "quantdemoire_weight",
        "quantsr_act",
        "quantsr_weight",
    }
    if (
        method == "percentile"
        and sample_size is not None
        and sample_size > 0
        and tensor.numel() > MAX_DIRECT_PERCENTILE_NUMEL
    ):
        use_sampling = True
    samples = sample_tensor_values(
        tensor,
        sample_size=sample_size if use_sampling else None,
        sample_rate=sample_rate if use_sampling else None,
    )
    # A single observation cannot define an activation interval. This optional
    # guard is an explicit reproduction assumption, not an inferred paper rule.
    pooled=tensor.ndim==4 and tensor.shape[-2:]==(1,1)
    if (small_tensor_policy in {"full_if_single", "full_if_pooled"} and samples.numel()==1 and tensor.numel()>1
        or small_tensor_policy=='full_if_pooled' and pooled):
        samples = tensor.detach().reshape(-1)

    if method in {"quantsr_act", "quantsr_weight"}:
        signed = method == "quantsr_weight" or bool(samples.min() < -1e-5)
        _, qmax = get_qrange(num_bits, signed=signed)
        alpha = (2.0 * samples.abs().mean() / math.sqrt(qmax)).clamp_min(1e-8)
        lower = -alpha * qmax if signed else torch.zeros_like(alpha)
        upper = alpha * qmax
    elif method in {
        "minmax",
        "sample_minmax",
    }:
        lower = samples.min()
        upper = samples.max()
    elif method in {
        "percentile",
        "sample_percentile",
        "quantdemoire_act",
        "quantdemoire_weight",
    }:
        lower_q, upper_q = _clamp_percentile_bounds(lower_percentile, upper_percentile)
        lower = torch.quantile(samples, lower_q / 100.0)
        upper = torch.quantile(samples, upper_q / 100.0)
    else:
        raise ValueError(f"Unsupported observer method: {method}")

    if float(lower) == float(upper):
        upper = upper + _as_tensor(1e-6, upper)
    return lower, upper


def get_qrange(num_bits, symmetric=False, signed=None):
    if signed is None:
        signed = bool(symmetric)
    if signed:
        qmin = -(2 ** (num_bits - 1))
        qmax = 2 ** (num_bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** num_bits - 1
    return qmin, qmax


def calculate_qparams_from_bounds(lower, upper, num_bits, symmetric=False, signed=None, eps=1e-8):
    lower = torch.as_tensor(lower)
    upper = torch.as_tensor(upper, device=lower.device, dtype=lower.dtype)
    qmin, qmax = get_qrange(num_bits, symmetric=symmetric, signed=signed)

    if symmetric:
        max_abs = torch.maximum(lower.abs(), upper.abs()).clamp_min(eps)
        scale = max_abs / float(qmax)
        zero_point = torch.zeros_like(scale)
    else:
        scale = (upper - lower).clamp_min(eps) / float(qmax - qmin)
        zero_point = ste_round(_as_tensor(qmin, scale) - lower / scale)
        zero_point = zero_point.clamp(qmin, qmax)

    return scale, zero_point, qmin, qmax


def fake_quantize_from_qparams(tensor, scale, zero_point, qmin, qmax):
    scale = _as_tensor(scale, tensor).clamp_min(1e-8)
    zero_point = _as_tensor(zero_point, tensor)
    quantized = ste_round(tensor / scale + zero_point)
    quantized = quantized.clamp(qmin, qmax)
    return (quantized - zero_point) * scale


def fake_quantize_from_bounds(tensor, lower, upper, num_bits, symmetric=False, signed=None):
    scale, zero_point, qmin, qmax = calculate_qparams_from_bounds(
        lower,
        upper,
        num_bits=num_bits,
        symmetric=symmetric,
        signed=signed,
    )
    return fake_quantize_from_qparams(tensor, scale, zero_point, qmin, qmax)


def fake_quantize_piecewise_symmetric(
    tensor,
    breakpoint,
    upper,
    num_bits,
    dense_positive_bins=4,
    eps=1e-8,
):
    if num_bits < 2:
        raise ValueError("piecewise symmetric quantization requires at least 2 bits")

    positive_bins = 2 ** (num_bits - 1) - 1
    dense_positive_bins = int(max(1, min(dense_positive_bins, positive_bins - 1)))
    outlier_positive_bins = positive_bins - dense_positive_bins
    if outlier_positive_bins <= 0:
        raise ValueError("piecewise symmetric quantization needs at least one outlier bin")

    breakpoint = _as_tensor(breakpoint, tensor).abs().clamp_min(eps)
    upper = _as_tensor(upper, tensor).abs().clamp_min(eps)
    upper = torch.maximum(upper, breakpoint + _as_tensor(eps, tensor))

    abs_tensor = tensor.abs()
    sign = tensor.sign()
    dense_scale = breakpoint / float(dense_positive_bins)
    outlier_scale = (upper - breakpoint).clamp_min(eps) / float(outlier_positive_bins)

    dense_mask = abs_tensor <= breakpoint
    dense_q = ste_round(abs_tensor / dense_scale).clamp(0, dense_positive_bins)
    dense_deq = dense_q * dense_scale

    outlier_q = ste_round((abs_tensor - breakpoint) / outlier_scale).clamp(0, outlier_positive_bins)
    outlier_q = torch.where(abs_tensor > breakpoint, torch.clamp(outlier_q, min=1), outlier_q)
    outlier_deq = breakpoint + outlier_q * outlier_scale

    dequantized = torch.where(dense_mask, dense_deq, outlier_deq)
    return dequantized * sign


def channelwise_weight_qparams(weight, num_bits, symmetric=True, eps=1e-8):
    weight = weight.detach()
    flattened = weight.reshape(weight.size(0), -1)
    if symmetric:
        max_abs = flattened.abs().amax(dim=1, keepdim=True).clamp_min_(eps)
        qmin, qmax = get_qrange(num_bits, symmetric=True)
        scale = max_abs / float(qmax)
        zero_point = torch.zeros_like(scale)
    else:
        lower = flattened.min(dim=1, keepdim=True).values
        upper = flattened.max(dim=1, keepdim=True).values
        qmin, qmax = get_qrange(num_bits, symmetric=False)
        scale = (upper - lower).clamp_min_(eps) / float(qmax - qmin)
        zero_point = ste_round(torch.full_like(scale, qmin) - lower / scale).clamp_(qmin, qmax)

    shape = (weight.size(0),) + (1,) * (weight.ndim - 1)
    return scale.view(shape), zero_point.view(shape), qmin, qmax


def percentile_mask(tensor, lower_percentile, upper_percentile):
    lower_q, upper_q = _clamp_percentile_bounds(lower_percentile, upper_percentile)
    flat = tensor.detach().reshape(-1)
    lower = torch.quantile(flat, lower_q / 100.0)
    upper = torch.quantile(flat, upper_q / 100.0)
    return (tensor < lower) | (tensor > upper), lower, upper


def build_radial_mask(height, width, low_ratio, high_ratio, device):
    y = torch.linspace(-1.0, 1.0, steps=height, device=device)
    x = torch.linspace(0.0, 1.0, steps=(width // 2) + 1, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    low = (radius <= low_ratio).float()
    mid = ((radius > low_ratio) & (radius <= high_ratio)).float()
    high = (radius > high_ratio).float()
    return low, mid, high
