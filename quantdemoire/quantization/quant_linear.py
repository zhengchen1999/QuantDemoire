"""Fake-quantized linear layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .observers import (
    calculate_qparams_from_bounds,
    estimate_range,
    fake_quantize_from_bounds,
    fake_quantize_from_qparams,
    percentile_mask,
    sample_tensor_values,
)


def _cfg(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class QuantizedLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        quant_config=None,
        layer_name="",
        layer_index=-1,
        skip_quantization=False,
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.layer_name = layer_name
        self.layer_index = layer_index
        self.skip_quantization = bool(skip_quantization)
        self.calibration_mode = 0

        self.nbits_w = int(_cfg(quant_config, "nbits_w", 4))
        self.nbits_a = int(_cfg(quant_config, "nbits_a", 4))
        self.weight_observer = str(_cfg(quant_config, "weight_observer", "minmax")).lower()
        self.act_observer = str(_cfg(quant_config, "act_observer", "minmax")).lower()
        self.weight_symmetric = bool(_cfg(quant_config, "weight_symmetric", True))
        self.act_symmetric = bool(_cfg(quant_config, "act_symmetric", False))
        self.use_smooth_scale = bool(_cfg(quant_config, "use_smooth_scale", False))
        self.smooth_alpha = float(_cfg(quant_config, "smooth_alpha", 0.5))
        self.input_channel_sample_size = int(_cfg(quant_config, "input_channel_sample_size", 1024))
        self.act_sample_size = int(_cfg(quant_config, "act_sample_size", 10000))
        self.act_percentile_low = float(_cfg(quant_config, "act_percentile_low", 0.1))
        self.act_percentile_high = float(_cfg(quant_config, "act_percentile_high", 99.9))
        self.weight_percentile_low = float(_cfg(quant_config, "weight_percentile_low", 0.5))
        self.weight_percentile_high = float(_cfg(quant_config, "weight_percentile_high", 99.5))
        self.enable_weight_outlier_fp16 = bool(_cfg(quant_config, "enable_weight_outlier_fp16", False))
        self.weight_outlier_mode = str(_cfg(quant_config, "weight_outlier_mode", "percentile_fp16")).lower()
        self.weight_outlier_percentile = float(_cfg(quant_config, "weight_outlier_percentile", 0.5))

        self.lb_a = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.ub_a = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.smooth_scale = nn.Parameter(torch.ones(in_features), requires_grad=False)

        self.register_buffer("original_weight", torch.empty_like(self.weight))
        self.register_buffer("quantized_weight", torch.empty_like(self.weight))
        self.register_buffer("weight_scale", torch.ones(out_features, 1))
        self.register_buffer("weight_zero_point", torch.zeros(out_features, 1))
        self.register_buffer("act_scale", torch.ones(1))
        self.register_buffer("act_zero_point", torch.zeros(1))
        self.register_buffer("input_absmax_sum", torch.zeros(in_features))
        self.register_buffer("input_absmax_batches", torch.zeros(1))
        self.register_buffer("act_lower_sum", torch.zeros(1))
        self.register_buffer("act_upper_sum", torch.zeros(1))
        self.register_buffer("act_batches", torch.zeros(1))
        self.register_buffer("outlier_mask", torch.zeros_like(self.weight, dtype=torch.bool))
        self.register_buffer("original_outliers", torch.zeros_like(self.weight))
        self.register_buffer("weights_quantized", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("activation_ready", torch.zeros(1, dtype=torch.bool))

        self.original_weight.copy_(self.weight.detach())
        self.quantized_weight.copy_(self.weight.detach())

    @classmethod
    def from_float_linear(
        cls,
        linear,
        quant_config=None,
        layer_name="",
        layer_index=-1,
        skip_quantization=False,
    ):
        quant_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            quant_config=quant_config,
            layer_name=layer_name,
            layer_index=layer_index,
            skip_quantization=skip_quantization,
        )
        quant_linear = quant_linear.to(device=linear.weight.device, dtype=linear.weight.dtype)
        quant_linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            quant_linear.bias.data.copy_(linear.bias.data)
        quant_linear.original_weight.copy_(linear.weight.detach())
        quant_linear.quantized_weight.copy_(linear.weight.detach())
        return quant_linear

    def set_calibration_mode(self, mode):
        self.calibration_mode = int(mode)

    def set_quant_trainable(self, trainable=False, train_smooth=True):
        self.lb_a.requires_grad_(bool(trainable))
        self.ub_a.requires_grad_(bool(trainable))
        self.smooth_scale.requires_grad_(bool(trainable and train_smooth and self.use_smooth_scale))

    def reset_calibration_buffers(self):
        self.input_absmax_sum.zero_()
        self.input_absmax_batches.zero_()
        self.act_lower_sum.zero_()
        self.act_upper_sum.zero_()
        self.act_batches.zero_()
        self.activation_ready.zero_()

    def reset_quantization_state(self):
        self.reset_calibration_buffers()
        self.weights_quantized.zero_()
        self.quantized_weight = self.original_weight.detach().clone()
        self.outlier_mask = torch.zeros_like(self.weight, dtype=torch.bool)
        self.original_outliers = torch.zeros_like(self.weight)
        self.lb_a.data.zero_()
        self.ub_a.data.zero_()
        self.smooth_scale.data.fill_(1.0)

    def _collect_pass1_stats(self, x):
        samples = x.detach().reshape(-1, x.shape[-1])
        if self.input_channel_sample_size is not None and self.input_channel_sample_size > 0 and samples.size(0) > self.input_channel_sample_size:
            indices = torch.randint(0, samples.size(0), (self.input_channel_sample_size,), device=samples.device)
            samples = samples[indices]
        channel_absmax = samples.abs().amax(dim=0).clamp_min_(1e-6)
        self.input_absmax_sum.add_(channel_absmax)
        self.input_absmax_batches.add_(1)

    def compute_smooth_scale(self):
        if self.skip_quantization or not self.use_smooth_scale:
            self.smooth_scale.data.fill_(1.0)
            return self.smooth_scale
        if int(self.input_absmax_batches.item()) == 0:
            return self.smooth_scale

        avg_absmax = self.input_absmax_sum / self.input_absmax_batches.clamp_min(1.0)
        weight_absmax = self.original_weight.detach().abs().amax(dim=0).clamp_min_(1e-6)
        scale = (avg_absmax.pow(self.smooth_alpha) / weight_absmax.pow(1.0 - self.smooth_alpha)).clamp_min_(1e-6)
        self.smooth_scale.data.copy_(scale)
        return self.smooth_scale

    def _apply_smooth_scale(self, x):
        if self.skip_quantization or not self.use_smooth_scale:
            return x
        return x / self.smooth_scale.view(*([1] * (x.ndim - 1)), -1)

    def _collect_pass2_stats(self, x):
        observed = self._apply_smooth_scale(x)
        lower, upper = estimate_range(
            observed,
            method=self.act_observer,
            sample_size=self.act_sample_size,
            lower_percentile=self.act_percentile_low,
            upper_percentile=self.act_percentile_high,
        )
        self.act_lower_sum.add_(lower.detach().view(1))
        self.act_upper_sum.add_(upper.detach().view(1))
        self.act_batches.add_(1)

    def finalize_activation_params(self):
        if self.skip_quantization:
            self.lb_a.data.zero_()
            self.ub_a.data.zero_()
            self.activation_ready.fill_(True)
            return

        if int(self.act_batches.item()) == 0:
            return

        lower = self.act_lower_sum / self.act_batches.clamp_min(1.0)
        upper = self.act_upper_sum / self.act_batches.clamp_min(1.0)
        if torch.allclose(lower, upper):
            upper = upper + upper.new_tensor(1e-6)

        self.lb_a.data.copy_(lower.squeeze())
        self.ub_a.data.copy_(upper.squeeze())
        scale, zero_point, _, _ = calculate_qparams_from_bounds(
            lower.squeeze(),
            upper.squeeze(),
            num_bits=self.nbits_a,
            symmetric=self.act_symmetric,
        )
        self.act_scale.copy_(scale.view(1))
        self.act_zero_point.copy_(zero_point.view(1))
        self.activation_ready.fill_(True)

    def _weight_source(self):
        weight = self.original_weight.detach().clone()
        if self.use_smooth_scale and not self.skip_quantization:
            weight = weight * self.smooth_scale.detach().view(1, -1)
        return weight

    def _quantize_weight_row(self, weight_row, non_outlier_row):
        lower, upper = estimate_range(
            non_outlier_row,
            method=self.weight_observer if self.weight_observer != "ocs" else "minmax",
            sample_size=None,
            lower_percentile=self.weight_percentile_low,
            upper_percentile=self.weight_percentile_high,
        )
        scale, zero_point, qmin, qmax = calculate_qparams_from_bounds(
            lower,
            upper,
            num_bits=self.nbits_w,
            symmetric=self.weight_symmetric,
        )
        quantized = fake_quantize_from_qparams(weight_row, scale, zero_point, qmin, qmax)
        return quantized, scale, zero_point

    def _quantize_weight_tensor(self, weight):
        quantized = torch.empty_like(weight)
        scales = torch.empty((weight.size(0), 1), device=weight.device, dtype=weight.dtype)
        zero_points = torch.empty((weight.size(0), 1), device=weight.device, dtype=weight.dtype)
        outlier_mask = torch.zeros_like(weight, dtype=torch.bool)
        original_outliers = torch.zeros_like(weight)

        if self.enable_weight_outlier_fp16 and self.weight_outlier_mode != "none":
            mask, _, _ = percentile_mask(
                weight,
                lower_percentile=self.weight_outlier_percentile,
                upper_percentile=100.0 - self.weight_outlier_percentile,
            )
            outlier_mask = mask
            original_outliers = weight.clone()

        for row_idx in range(weight.size(0)):
            row = weight[row_idx]
            if outlier_mask.any():
                non_outlier = row[~outlier_mask[row_idx]]
                if non_outlier.numel() == 0:
                    non_outlier = row.reshape(-1)
            else:
                non_outlier = row.reshape(-1)

            quantized_row, scale, zero_point = self._quantize_weight_row(row, non_outlier)
            if outlier_mask.any():
                quantized_row = torch.where(
                    outlier_mask[row_idx],
                    original_outliers[row_idx],
                    quantized_row,
                )
            quantized[row_idx] = quantized_row
            scales[row_idx] = scale.reshape(1)
            zero_points[row_idx] = zero_point.reshape(1)

        return quantized, scales, zero_points, outlier_mask, original_outliers

    def quantize_weights(self):
        if self.skip_quantization:
            self.quantized_weight = self.original_weight.detach().clone()
            self.weights_quantized.fill_(True)
            self.activation_ready.fill_(True)
            return

        weight = self._weight_source()
        quantized, scales, zero_points, outlier_mask, original_outliers = self._quantize_weight_tensor(weight)
        self.quantized_weight = quantized.detach().clone()
        self.weight_scale = scales.detach().clone()
        self.weight_zero_point = zero_points.detach().clone()
        self.outlier_mask = outlier_mask.detach().clone()
        self.original_outliers = original_outliers.detach().clone()
        self.weights_quantized.fill_(True)

    def export_quant_state(self):
        return {
            "layer_name": self.layer_name,
            "layer_index": self.layer_index,
            "skip_quantization": self.skip_quantization,
            "lb_a": self.lb_a.detach().cpu(),
            "ub_a": self.ub_a.detach().cpu(),
            "smooth_scale": self.smooth_scale.detach().cpu(),
            "weight_scale": self.weight_scale.detach().cpu(),
            "weight_zero_point": self.weight_zero_point.detach().cpu(),
            "quantized_weight": self.quantized_weight.detach().cpu(),
            "outlier_mask": self.outlier_mask.detach().cpu(),
            "original_outliers": self.original_outliers.detach().cpu(),
            "weights_quantized": bool(self.weights_quantized.item()),
            "activation_ready": bool(self.activation_ready.item()),
        }

    def load_quant_state(self, state):
        self.layer_name = state.get("layer_name", self.layer_name)
        self.layer_index = state.get("layer_index", self.layer_index)
        self.skip_quantization = bool(state.get("skip_quantization", self.skip_quantization))
        self.lb_a.data.copy_(state["lb_a"].to(self.lb_a.device, self.lb_a.dtype))
        self.ub_a.data.copy_(state["ub_a"].to(self.ub_a.device, self.ub_a.dtype))
        self.smooth_scale.data.copy_(state["smooth_scale"].to(self.smooth_scale.device, self.smooth_scale.dtype))
        self.weight_scale = state["weight_scale"].to(self.weight_scale.device, self.weight_scale.dtype)
        self.weight_zero_point = state["weight_zero_point"].to(self.weight_zero_point.device, self.weight_zero_point.dtype)
        self.quantized_weight = state["quantized_weight"].to(self.weight.device, self.weight.dtype)
        self.outlier_mask = state["outlier_mask"].to(self.weight.device)
        self.original_outliers = state["original_outliers"].to(self.weight.device, self.weight.dtype)
        self.weights_quantized.fill_(bool(state.get("weights_quantized", True)))
        self.activation_ready.fill_(bool(state.get("activation_ready", True)))

    def _float_forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def _quantized_forward(self, x):
        x_q = self._apply_smooth_scale(x)
        x_q = fake_quantize_from_bounds(
            x_q,
            self.lb_a,
            self.ub_a,
            num_bits=self.nbits_a,
            symmetric=self.act_symmetric,
        )
        return F.linear(x_q, self.quantized_weight, self.bias)

    def forward(self, x):
        if self.skip_quantization:
            return self._float_forward(x)

        if self.calibration_mode == 2:
            with torch.no_grad():
                self._collect_pass1_stats(x)
            return self._float_forward(x)

        if self.calibration_mode == 1:
            with torch.no_grad():
                self._collect_pass2_stats(x)
            return self._float_forward(x)

        if not bool(self.weights_quantized.item()) or not bool(self.activation_ready.item()):
            return self._float_forward(x)

        return self._quantized_forward(x)
