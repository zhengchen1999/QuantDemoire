"""Fake-quantization building blocks and calibration pipeline."""

from .calibration import run_two_pass_calibration
from .frequency_loss import build_frequency_loss, build_reconstruction_loss
from .pipeline import (
    collect_trainable_quant_params,
    enable_frequency_calibration,
    finalize_activation_ranges,
    get_quantized_linears,
    get_quantized_convs,
    get_quantized_modules,
    load_quant_state,
    quantize_model_weights,
    replace_model_convs,
    replace_model_modules,
    resolve_frequency_config,
    resolve_quant_config,
    save_quant_state,
    set_calibration_mode,
)
from .quant_conv import QuantizedConv2d
from .quant_linear import QuantizedLinear
from .replace import replace_conv_modules, replace_linear_modules, replace_quantizable_modules

__all__ = [
    "QuantizedConv2d",
    "QuantizedLinear",
    "build_frequency_loss",
    "build_reconstruction_loss",
    "collect_trainable_quant_params",
    "enable_frequency_calibration",
    "finalize_activation_ranges",
    "get_quantized_convs",
    "get_quantized_linears",
    "get_quantized_modules",
    "load_quant_state",
    "quantize_model_weights",
    "replace_conv_modules",
    "replace_linear_modules",
    "replace_quantizable_modules",
    "replace_model_convs",
    "replace_model_modules",
    "resolve_frequency_config",
    "resolve_quant_config",
    "run_two_pass_calibration",
    "save_quant_state",
    "set_calibration_mode",
]
