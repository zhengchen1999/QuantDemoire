"""Model replacement, calibration, and state persistence helpers."""

import json
import os

import torch

from .quant_conv import QuantizedConv2d
from .quant_linear import QuantizedLinear
from .replace import replace_conv_modules, replace_quantizable_modules


def _get_attr(source, name, default=None):
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _first_available(source, names, default=None):
    for name in names:
        value = _get_attr(source, name, None)
        if value is not None:
            return value
    return default


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _load_activation_sign_map(path):
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle) or {}
    if isinstance(payload, dict) and isinstance(payload.get("layers"), dict):
        return payload["layers"]
    if isinstance(payload, dict):
        return payload
    return {}


def resolve_quant_config(args):
    method = str(_first_available(args, ["METHOD", "QUANT_METHOD"], "quantdemoire")).lower()
    default_weight_observer = {
        "minmax": "minmax",
        "percentile": "percentile",
        "quantdemoire": "quantdemoire_weight",
        "quantsr": "quantsr_weight",
        "ptq4sr": "percentile",
        "repquant": "minmax",
        "wang2025": "minmax",
        "oaptqsr": "minmax",
        "ocs": "ocs",
    }.get(method, "minmax")
    default_act_observer = {
        "minmax": "minmax",
        "percentile": "percentile",
        "quantdemoire": "quantdemoire_act",
        "quantsr": "quantsr_act",
        "ptq4sr": "percentile",
        "repquant": "minmax",
        "wang2025": "wang2025_plq",
        "oaptqsr": "wang2025_plq",
        "ocs": "minmax",
    }.get(method, "minmax")

    activation_sign_state_path = _first_available(args, ["ACTIVATION_SIGN_STATE_PATH"], "")
    return {
        "method": method,
        "nbits_w": int(_first_available(args, ["N_BITS_W", "QUANT_BITS_W", "QUANTIZE_BIT"], 4)),
        "nbits_a": int(_first_available(args, ["N_BITS_A", "QUANT_BITS_A", "QUANTIZE_BIT"], 4)),
        "weight_observer": str(_first_available(args, ["WEIGHT_OBSERVER"], default_weight_observer)).lower(),
        "act_observer": str(_first_available(args, ["ACT_OBSERVER"], default_act_observer)).lower(),
        "weight_symmetric": _as_bool(_first_available(args, ["WEIGHT_SYMMETRIC"], True)),
        "act_symmetric": _as_bool(_first_available(args, ["ACT_SYMMETRIC"], False)),
        "use_smooth_scale": _as_bool(
            _first_available(args, ["USE_SMOOTH_SCALE"], method in {"quantdemoire", "repquant"})
        ),
        "use_activation_shift": _as_bool(
            _first_available(args, ["USE_ACTIVATION_SHIFT"], method == "quantsr")
        ),
        "learn_weight_bounds": _as_bool(
            _first_available(args, ["LEARN_WEIGHT_BOUNDS"], method in {"quantsr", "ptq4sr"})
        ),
        "smooth_alpha": float(_first_available(args, ["SMOOTH_ALPHA"], 0.5)),
        "input_channel_sample_size": int(_first_available(args, ["INPUT_CHANNEL_SAMPLE_SIZE"], 1024)),
        "act_sample_size": int(_first_available(args, ["ACT_SAMPLE_SIZE"], 10000)),
        "input_channel_sample_rate": float(_first_available(args, ["INPUT_CHANNEL_SAMPLE_RATE"], 0.0)),
        "act_sample_rate": float(_first_available(args, ["ACT_SAMPLE_RATE"], 0.0)),
        "act_small_tensor_policy": str(_first_available(args, ["ACT_SMALL_TENSOR_POLICY"], "legacy")),
        "calibration_aggregation": str(_first_available(args, ["CALIBRATION_AGGREGATION"], "mean")),
        "act_percentile_low": float(_first_available(args, ["ACT_PERCENTILE_LOW"], 0.1)),
        "act_percentile_high": float(_first_available(args, ["ACT_PERCENTILE_HIGH"], 99.9)),
        "weight_percentile_low": float(_first_available(args, ["WEIGHT_PERCENTILE_LOW"], 0.5)),
        "weight_percentile_high": float(_first_available(args, ["WEIGHT_PERCENTILE_HIGH"], 99.5)),
        "enable_weight_outlier_fp16": _as_bool(_first_available(args, ["ENABLE_WEIGHT_OUTLIER_FP16"], method == "quantdemoire")),
        "weight_outlier_mode": str(_first_available(args, ["WEIGHT_OUTLIER_MODE"], "percentile_fp16")).lower(),
        "weight_outlier_percentile": float(_first_available(args, ["WEIGHT_OUTLIER_PERCENTILE"], 0.5)),
        "act_breakpoint_percentile": float(_first_available(args, ["ACT_BREAKPOINT_PERCENTILE"], 99.0)),
        "act_dense_positive_bins": int(_first_available(args, ["ACT_DENSE_POSITIVE_BINS"], 4)),
        "skip_first_conv": _as_bool(_first_available(args, ["SKIP_FIRST_CONV"], True)),
        "skip_last_conv": _as_bool(_first_available(args, ["SKIP_LAST_CONV"], True)),
        "skip_first_linear": _as_bool(_first_available(args, ["SKIP_FIRST_LINEAR"], False)),
        "skip_last_linear": _as_bool(_first_available(args, ["SKIP_LAST_LINEAR"], False)),
        "include_patterns": _as_list(_first_available(args, ["INCLUDE_PATTERNS"], [])),
        "exclude_patterns": _as_list(_first_available(args, ["EXCLUDE_PATTERNS"], [])),
        "use_unsigned_activation_for_nonnegative": _as_bool(
            _first_available(args, ["USE_UNSIGNED_ACTIVATION_FOR_NONNEGATIVE"], True)
        ),
        "activation_sign_state_path": activation_sign_state_path,
        "activation_sign_map": _load_activation_sign_map(activation_sign_state_path),
    }


def resolve_frequency_config(args):
    return {
        "enable": _as_bool(_first_available(args, ["FREQUENCY_ENABLE"], False)),
        "finetune_enable": _as_bool(_first_available(args, ["FINETUNE_ENABLE"], False)),
        "alternate_finetune": _as_bool(_first_available(args, ["ALTERNATE_FINETUNE"], False)),
        "loss_type": str(_first_available(args, ["FREQUENCY_LOSS_TYPE"], "gaussian")).lower(),
        "gaussian_levels": int(_first_available(args, ["GAUSSIAN_LEVELS"], 3)),
        "gaussian_first_dilation": int(_first_available(args, ['GAUSSIAN_FIRST_DILATION'], 1)),
        "aux_full_l1_weight": float(_first_available(args, ["AUX_FULL_L1_WEIGHT"], 0.1)),
        "perceptual_weight": float(_first_available(args, ["FAC_PERCEPTUAL_WEIGHT"], 0.0)),
        "perceptual_preprocess": str(_first_available(args, ['FAC_PERCEPTUAL_PREPROCESS'], 'legacy_clamped')),
        "perceptual_feature_indices": _as_list(
            _first_available(args, ["FAC_PERCEPTUAL_FEATURE_INDICES"], [15])
        ),
        "reconstruction_l1_weight": float(_first_available(args, ["RECONSTRUCTION_L1_WEIGHT"], 1.0)),
        "fft_low_ratio": float(_first_available(args, ["FFT_LOW_RATIO"], 0.15)),
        "fft_high_ratio": float(_first_available(args, ["FFT_HIGH_RATIO"], 0.45)),
        "fft_low_weight": float(_first_available(args, ["FFT_LOW_WEIGHT"], 1.0)),
        "fft_mid_weight": float(_first_available(args, ["FFT_MID_WEIGHT"], 1.0)),
        "fft_high_weight": float(_first_available(args, ["FFT_HIGH_WEIGHT"], 0.0)),
        "fft_use_log_mag": _as_bool(_first_available(args, ["FFT_USE_LOG_MAG"], True)),
        "wavelet_levels": int(_first_available(args, ["WAVELET_LEVELS"], 3)),
        "wavelet_low_weight": float(_first_available(args, ["WAVELET_LOW_WEIGHT"], 1.0)),
        "wavelet_detail_weight": float(_first_available(args, ["WAVELET_DETAIL_WEIGHT"], 1.0)),
        "epochs": int(_first_available(args, ["FREQUENCY_EPOCHS"], 5)),
        "lr": float(_first_available(args, ["FREQUENCY_LR"], 5e-4)),
        "weight_lr": float(_first_available(args, ["FREQUENCY_WEIGHT_LR"], -1.0)),
        "scheduler": str(_first_available(args, ["FREQUENCY_SCHEDULER"], "none")).lower(),
        "scheduler_t0_epochs": float(_first_available(args, ["FREQUENCY_SCHEDULER_T0_EPOCHS"], 1.0)),
        "scheduler_t_mult": int(_first_available(args, ["FREQUENCY_SCHEDULER_T_MULT"], 1)),
        "scheduler_eta_min": float(_first_available(args, ["FREQUENCY_SCHEDULER_ETA_MIN"], 1e-6)),
        "train_smooth": _as_bool(_first_available(args, ["TRAIN_SMOOTH"], True)),
    }


def get_quantized_convs(model):
    return [module for module in model.modules() if isinstance(module, QuantizedConv2d)]


def get_quantized_linears(model):
    return [module for module in model.modules() if isinstance(module, QuantizedLinear)]


def get_quantized_modules(model):
    return [module for module in model.modules() if isinstance(module, (QuantizedConv2d, QuantizedLinear))]


def replace_model_convs(model, args):
    quant_config = resolve_quant_config(args)
    return replace_conv_modules(model, quant_config=quant_config)


def replace_model_modules(model, args):
    quant_config = resolve_quant_config(args)
    return replace_quantizable_modules(model, quant_config=quant_config)


def set_calibration_mode(model, mode):
    for module in get_quantized_modules(model):
        module.set_calibration_mode(mode)


def reset_quantization_state(model):
    for module in get_quantized_modules(model):
        module.reset_quantization_state()


def compute_smooth_scales(model):
    for module in get_quantized_modules(model):
        module.compute_smooth_scale()


def finalize_activation_ranges(model):
    for module in get_quantized_modules(model):
        module.finalize_activation_params()


def quantize_model_weights(model):
    for module in get_quantized_modules(model):
        module.quantize_weights()


def enable_frequency_calibration(model, trainable=True, train_smooth=True):
    for module in get_quantized_modules(model):
        module.set_quant_trainable(trainable=trainable, train_smooth=train_smooth)


def project_quantization_bounds(model, min_width=1e-6):
    for module in get_quantized_modules(model):
        module.project_quantization_bounds(min_width=min_width)


def collect_trainable_quant_params(model, train_smooth=True):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    params = []
    for module in get_quantized_modules(model):
        module.set_quant_trainable(trainable=True, train_smooth=train_smooth)
        if module.lb_a.requires_grad:
            params.append(module.lb_a)
        if module.ub_a.requires_grad:
            params.append(module.ub_a)
        if module.smooth_scale.requires_grad:
            params.append(module.smooth_scale)
        activation_shift = getattr(module, "activation_shift", None)
        if activation_shift is not None and activation_shift.requires_grad:
            params.append(activation_shift)
        for name in ("lb_w", "ub_w"):
            parameter = getattr(module, name, None)
            if parameter is not None and parameter.requires_grad:
                params.append(parameter)
    return params


def save_quant_state(model, path):
    payload = {}
    for name, module in model.named_modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            payload[name] = module.export_quant_state()
    torch.save(payload, path)


def load_quant_state(model, path, strict=True):
    payload = torch.load(path, map_location="cpu")
    loaded = set()
    for name, module in model.named_modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            if name not in payload:
                if strict:
                    raise KeyError(f"Missing quant state for layer: {name}")
                continue
            module.load_quant_state(payload[name])
            loaded.add(name)
    if strict:
        missing = set(payload.keys()) - loaded
        if missing:
            raise KeyError(f"Unused quant state entries: {sorted(missing)}")
    return payload
