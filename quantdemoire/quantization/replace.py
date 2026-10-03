"""Utilities for replacing float modules with fake-quantized modules."""

import torch.nn as nn

from .quant_conv import QuantizedConv2d
from .quant_linear import QuantizedLinear


def _cfg(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _matches_any(name, patterns):
    if not patterns:
        return False
    return any(pattern in name for pattern in patterns)


def _module_entries(model, target_type):
    entries = []
    for module_name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, target_type):
                full_name = f"{module_name}.{child_name}" if module_name else child_name
                entries.append((module, child_name, full_name, child))
    return entries


def should_replace_module(layer_name, layer_index, num_layers, quant_config, skip_first_key, skip_last_key):
    if _matches_any(layer_name, _cfg(quant_config, "exclude_patterns", [])):
        return False

    include_patterns = _cfg(quant_config, "include_patterns", [])
    if include_patterns and not _matches_any(layer_name, include_patterns):
        return False

    skip_first = bool(_cfg(quant_config, skip_first_key, False))
    skip_last = bool(_cfg(quant_config, skip_last_key, False))
    if skip_first and layer_index == 0:
        return False
    if skip_last and layer_index == num_layers - 1:
        return False
    return True


def replace_module(parent, child_name, new_module):
    setattr(parent, child_name, new_module)


def build_quant_conv_from_conv(conv, quant_config, layer_name, layer_index, skip_quantization=False):
    quant_conv = QuantizedConv2d.from_float_conv(
        conv,
        quant_config=quant_config,
        layer_name=layer_name,
        layer_index=layer_index,
        skip_quantization=skip_quantization,
    )
    activation_sign_map = _cfg(quant_config, "activation_sign_map", {})
    if activation_sign_map and hasattr(quant_conv, "set_activation_sign_mode"):
        layer_info = activation_sign_map.get(layer_name)
        if isinstance(layer_info, dict):
            unsigned_activation = bool(layer_info.get("non_negative", False))
        else:
            unsigned_activation = bool(layer_info)
        quant_conv.set_activation_sign_mode(unsigned_activation)
    return quant_conv


def build_quant_linear_from_linear(linear, quant_config, layer_name, layer_index, skip_quantization=False):
    return QuantizedLinear.from_float_linear(
        linear,
        quant_config=quant_config,
        layer_name=layer_name,
        layer_index=layer_index,
        skip_quantization=skip_quantization,
    )


def replace_conv_modules(model, quant_config=None):
    entries = _module_entries(model, nn.Conv2d)
    num_layers = len(entries)
    replaced = []

    for layer_index, (parent, child_name, layer_name, conv) in enumerate(entries):
        replace = should_replace_module(
            layer_name,
            layer_index,
            num_layers,
            quant_config,
            skip_first_key="skip_first_conv",
            skip_last_key="skip_last_conv",
        )
        quant_conv = build_quant_conv_from_conv(
            conv,
            quant_config=quant_config,
            layer_name=layer_name,
            layer_index=layer_index,
            skip_quantization=not replace,
        )
        replace_module(parent, child_name, quant_conv)
        replaced.append(quant_conv)

    return replaced


def replace_linear_modules(model, quant_config=None):
    entries = _module_entries(model, nn.Linear)
    num_layers = len(entries)
    replaced = []

    for layer_index, (parent, child_name, layer_name, linear) in enumerate(entries):
        replace = should_replace_module(
            layer_name,
            layer_index,
            num_layers,
            quant_config,
            skip_first_key="skip_first_linear",
            skip_last_key="skip_last_linear",
        )
        quant_linear = build_quant_linear_from_linear(
            linear,
            quant_config=quant_config,
            layer_name=layer_name,
            layer_index=layer_index,
            skip_quantization=not replace,
        )
        replace_module(parent, child_name, quant_linear)
        replaced.append(quant_linear)

    return replaced


def replace_quantizable_modules(model, quant_config=None):
    replaced = []
    replaced.extend(replace_conv_modules(model, quant_config=quant_config))
    replaced.extend(replace_linear_modules(model, quant_config=quant_config))
    return replaced
