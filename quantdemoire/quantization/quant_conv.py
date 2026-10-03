"""Fake-quantized convolution layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .observers import (
    calculate_qparams_from_bounds,
    estimate_channel_absmax,
    estimate_range,
    fake_quantize_from_bounds,
    fake_quantize_piecewise_symmetric,
    fake_quantize_from_qparams,
    get_qrange,
    percentile_mask,
    rank_tail_mask,
)


def _cfg(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class QuantizedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        quant_config=None,
        layer_name="",
        layer_index=-1,
        skip_quantization=False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
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
        self.use_activation_shift = bool(_cfg(quant_config, "use_activation_shift", False))
        self.learn_weight_bounds = bool(_cfg(quant_config, "learn_weight_bounds", False))
        self.smooth_alpha = float(_cfg(quant_config, "smooth_alpha", 0.5))
        self.input_channel_sample_size = int(_cfg(quant_config, "input_channel_sample_size", 1024))
        self.act_sample_size = int(_cfg(quant_config, "act_sample_size", 10000))
        self.input_channel_sample_rate = float(_cfg(quant_config, "input_channel_sample_rate", 0.0))
        self.act_sample_rate = float(_cfg(quant_config, "act_sample_rate", 0.0))
        self.act_small_tensor_policy = str(_cfg(quant_config, "act_small_tensor_policy", "legacy"))
        self.calibration_aggregation = str(_cfg(quant_config, "calibration_aggregation", "mean"))
        if self.calibration_aggregation not in {"mean", "extrema"}:
            raise ValueError('calibration aggregation must be mean or extrema')
        self.act_percentile_low = float(_cfg(quant_config, "act_percentile_low", 0.1))
        self.act_percentile_high = float(_cfg(quant_config, "act_percentile_high", 99.9))
        self.weight_percentile_low = float(_cfg(quant_config, "weight_percentile_low", 0.5))
        self.weight_percentile_high = float(_cfg(quant_config, "weight_percentile_high", 99.5))
        self.enable_weight_outlier_fp16 = bool(_cfg(quant_config, "enable_weight_outlier_fp16", False))
        self.weight_outlier_mode = str(_cfg(quant_config, "weight_outlier_mode", "percentile_fp16")).lower()
        self.weight_outlier_percentile = float(_cfg(quant_config, "weight_outlier_percentile", 0.5))
        self.act_breakpoint_percentile = float(_cfg(quant_config, "act_breakpoint_percentile", 99.0))
        self.act_dense_positive_bins = int(_cfg(quant_config, "act_dense_positive_bins", 4))

        self.lb_a = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.ub_a = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.bp_a = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.smooth_scale = nn.Parameter(torch.ones(in_channels), requires_grad=False)
        self.activation_shift = nn.Parameter(torch.zeros(in_channels), requires_grad=False)
        self.lb_w = nn.Parameter(torch.zeros(out_channels, 1, 1, 1), requires_grad=False)
        self.ub_w = nn.Parameter(torch.zeros(out_channels, 1, 1, 1), requires_grad=False)

        self.register_buffer("original_weight", torch.empty_like(self.weight))
        self.register_buffer("quantized_weight", torch.empty_like(self.weight))
        self.register_buffer("weight_scale", torch.ones(out_channels, 1, 1, 1))
        self.register_buffer("weight_zero_point", torch.zeros(out_channels, 1, 1, 1))
        self.register_buffer("act_scale", torch.ones(1))
        self.register_buffer("act_zero_point", torch.zeros(1))
        self.register_buffer("input_absmax_sum", torch.zeros(in_channels))
        self.register_buffer("input_absmax_batches", torch.zeros(1))
        self.register_buffer("act_lower_sum", torch.zeros(1))
        self.register_buffer("act_upper_sum", torch.zeros(1))
        self.register_buffer("act_breakpoint_sum", torch.zeros(1))
        self.register_buffer("act_batches", torch.zeros(1))
        self.register_buffer("outlier_mask", torch.zeros_like(self.weight, dtype=torch.bool))
        self.register_buffer("original_outliers", torch.zeros_like(self.weight))
        self.register_buffer("weights_quantized", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("activation_ready", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("ocs_duplicate_index", torch.full((1,), -1, dtype=torch.long))

        self.original_weight.copy_(self.weight.detach())
        self.quantized_weight.copy_(self.weight.detach())

    @classmethod
    def from_float_conv(
        cls,
        conv,
        quant_config=None,
        layer_name="",
        layer_index=-1,
        skip_quantization=False,
    ):
        quant_conv = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            quant_config=quant_config,
            layer_name=layer_name,
            layer_index=layer_index,
            skip_quantization=skip_quantization,
        )
        quant_conv = quant_conv.to(device=conv.weight.device, dtype=conv.weight.dtype)
        quant_conv.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            quant_conv.bias.data.copy_(conv.bias.data)
        quant_conv.original_weight.copy_(conv.weight.detach())
        quant_conv.quantized_weight.copy_(conv.weight.detach())
        return quant_conv

    def set_layer_metadata(self, layer_name, layer_index, skip_quantization=None):
        self.layer_name = layer_name
        self.layer_index = layer_index
        if skip_quantization is not None:
            self.skip_quantization = bool(skip_quantization)

    def set_calibration_mode(self, mode):
        self.calibration_mode = int(mode)

    def set_quant_trainable(self, trainable=False, train_smooth=True):
        trainable = bool(trainable and not self.skip_quantization)
        self.lb_a.requires_grad_(bool(trainable))
        self.ub_a.requires_grad_(bool(trainable))
        self.smooth_scale.requires_grad_(bool(trainable and train_smooth and self.use_smooth_scale))
        self.activation_shift.requires_grad_(bool(trainable and self.use_activation_shift))
        self.lb_w.requires_grad_(bool(trainable and self.learn_weight_bounds))
        self.ub_w.requires_grad_(bool(trainable and self.learn_weight_bounds))

    @torch.no_grad()
    def project_quantization_bounds(self, min_width=1e-6):
        if self.skip_quantization:
            return
        lower = torch.minimum(self.lb_a, self.ub_a)
        upper = torch.maximum(self.lb_a, self.ub_a)
        upper = torch.maximum(upper, lower + upper.new_tensor(min_width))
        self.lb_a.copy_(lower)
        self.ub_a.copy_(upper)

        if self.learn_weight_bounds:
            lower_w = torch.minimum(self.lb_w, self.ub_w)
            upper_w = torch.maximum(self.lb_w, self.ub_w)
            upper_w = torch.maximum(upper_w, lower_w + upper_w.new_tensor(min_width))
            self.lb_w.copy_(lower_w)
            self.ub_w.copy_(upper_w)

    def reset_calibration_buffers(self):
        self.input_absmax_sum.zero_()
        self.input_absmax_batches.zero_()
        self.act_lower_sum.zero_()
        self.act_upper_sum.zero_()
        self.act_breakpoint_sum.zero_()
        self.act_batches.zero_()
        self.activation_ready.zero_()

    def reset_quantization_state(self):
        self.reset_calibration_buffers()
        self.weights_quantized.zero_()
        self.ocs_duplicate_index.fill_(-1)
        self.quantized_weight = self.original_weight.detach().clone()
        self.outlier_mask = torch.zeros_like(self.weight, dtype=torch.bool)
        self.original_outliers = torch.zeros_like(self.weight)
        self.lb_a.data.zero_()
        self.ub_a.data.zero_()
        self.bp_a.data.zero_()
        self.smooth_scale.data.fill_(1.0)
        self.activation_shift.data.zero_()
        self.lb_w.data.zero_()
        self.ub_w.data.zero_()

    def _collect_pass1_stats(self, x):
        channel_absmax = estimate_channel_absmax(
            x,
            sample_size=self.input_channel_sample_size,
            sample_rate=self.input_channel_sample_rate,
        )
        if self.calibration_aggregation == 'extrema':
            self.input_absmax_sum.copy_(torch.maximum(self.input_absmax_sum,channel_absmax.detach()))
        else:
            self.input_absmax_sum.add_(channel_absmax.detach())
        self.input_absmax_batches.add_(1)

    def compute_smooth_scale(self):
        if self.skip_quantization or not self.use_smooth_scale:
            self.smooth_scale.data.fill_(1.0)
            return self.smooth_scale
        if int(self.input_absmax_batches.item()) == 0:
            return self.smooth_scale

        avg_absmax = self.input_absmax_sum / self.input_absmax_batches.clamp_min(1.0)
        if self.calibration_aggregation == 'extrema':
            avg_absmax = self.input_absmax_sum
        weight_absmax = self.original_weight.detach().abs().amax(dim=(0, 2, 3)).clamp_min_(1e-6)
        scale = (avg_absmax.pow(self.smooth_alpha) / weight_absmax.pow(1.0 - self.smooth_alpha)).clamp_min_(1e-6)
        self.smooth_scale.data.copy_(scale)
        return self.smooth_scale

    def _apply_smooth_scale(self, x):
        if self.use_activation_shift and not self.skip_quantization:
            x = x + self.activation_shift.view(1, -1, 1, 1)
        if self.skip_quantization or not self.use_smooth_scale:
            return x
        return x / self.smooth_scale.view(1, -1, 1, 1)

    def _collect_pass2_stats(self, x):
        observed = self._apply_smooth_scale(x)
        lower, upper = estimate_range(
            observed,
            method="minmax" if self.act_observer in {"oaptqsr", "wang2025", "wang2025_plq"} else self.act_observer,
            sample_size=self.act_sample_size,
            sample_rate=self.act_sample_rate,
            lower_percentile=self.act_percentile_low,
            upper_percentile=self.act_percentile_high,
            num_bits=self.nbits_a,
            small_tensor_policy=self.act_small_tensor_policy,
        )
        if self.calibration_aggregation == 'extrema':
            if self.act_batches.item() == 0:
                self.act_lower_sum.copy_(lower.detach().view(1))
                self.act_upper_sum.copy_(upper.detach().view(1))
            else:
                self.act_lower_sum.copy_(torch.minimum(self.act_lower_sum,lower.detach().view(1)))
                self.act_upper_sum.copy_(torch.maximum(self.act_upper_sum,upper.detach().view(1)))
        else:
            self.act_lower_sum.add_(lower.detach().view(1))
            self.act_upper_sum.add_(upper.detach().view(1))
        if self.act_observer in {"oaptqsr", "wang2025", "wang2025_plq"}:
            abs_values = observed.detach().abs().reshape(-1)
            if self.act_sample_size > 0 and abs_values.numel() > self.act_sample_size:
                indices = torch.randint(0, abs_values.numel(), (self.act_sample_size,), device=abs_values.device)
                abs_values = abs_values[indices]
            breakpoint = torch.quantile(
                abs_values,
                self.act_breakpoint_percentile / 100.0,
            )
            self.act_breakpoint_sum.add_(breakpoint.detach().view(1))
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
        if self.calibration_aggregation == 'extrema':
            lower,upper = self.act_lower_sum,self.act_upper_sum
        if torch.allclose(lower, upper):
            upper = upper + upper.new_tensor(1e-6)

        self.lb_a.data.copy_(lower.squeeze())
        self.ub_a.data.copy_(upper.squeeze())
        if self.act_observer in {"oaptqsr", "wang2025", "wang2025_plq"}:
            breakpoint = self.act_breakpoint_sum / self.act_batches.clamp_min(1.0)
            breakpoint = torch.maximum(
                breakpoint.squeeze(),
                torch.tensor(1e-6, device=breakpoint.device, dtype=breakpoint.dtype),
            )
            self.bp_a.data.copy_(breakpoint)
        else:
            self.bp_a.data.zero_()
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
            weight = weight * self.smooth_scale.detach().view(1, -1, 1, 1)
        return weight

    def _quantize_weight_channel(self, weight_channel, non_outlier_channel):
        lower, upper = estimate_range(
            non_outlier_channel,
            method=self.weight_observer,
            sample_size=None,
            lower_percentile=self.weight_percentile_low,
            upper_percentile=self.weight_percentile_high,
            num_bits=self.nbits_w,
        )
        scale, zero_point, qmin, qmax = calculate_qparams_from_bounds(
            lower,
            upper,
            num_bits=self.nbits_w,
            symmetric=self.weight_symmetric,
        )
        quantized = fake_quantize_from_qparams(weight_channel, scale, zero_point, qmin, qmax)
        return quantized, scale, zero_point

    def _quantize_weight_tensor(self, weight):
        quantized = torch.empty_like(weight)
        scales = torch.empty((weight.size(0), 1, 1, 1), device=weight.device, dtype=weight.dtype)
        zero_points = torch.empty((weight.size(0), 1, 1, 1), device=weight.device, dtype=weight.dtype)
        outlier_mask = torch.zeros_like(weight, dtype=torch.bool)
        original_outliers = torch.zeros_like(weight)
        quant_source = weight

        if self.enable_weight_outlier_fp16 and self.weight_outlier_mode != "none":
            if self.weight_outlier_mode == 'rank_fp16':
                mask = rank_tail_mask(weight, self.weight_outlier_percentile/100)
            else:
                mask, lower, upper = percentile_mask(
                    weight,
                    lower_percentile=self.weight_outlier_percentile,
                    upper_percentile=100.0 - self.weight_outlier_percentile,
                )
            if self.weight_outlier_mode in ('percentile_fp16', 'rank_fp16'):
                outlier_mask = mask
                original_outliers = (weight.half().to(weight.dtype) if self.weight_outlier_mode=='rank_fp16'
                                     else weight.clone())
            elif self.weight_outlier_mode == "random_fp16":
                num_outliers = int(mask.sum().item())
                if num_outliers > 0:
                    flat_mask = torch.zeros(weight.numel(), device=weight.device, dtype=torch.bool)
                    random_indices = torch.randperm(weight.numel(), device=weight.device)[:num_outliers]
                    flat_mask[random_indices] = True
                    outlier_mask = flat_mask.view_as(weight)
                    original_outliers = weight.clone()
            elif self.weight_outlier_mode == "clip_percentile":
                quant_source = weight.clamp(min=lower, max=upper)

        for channel_idx in range(weight.size(0)):
            channel = quant_source[channel_idx]
            if outlier_mask.any():
                non_outlier = channel[~outlier_mask[channel_idx]]
                if non_outlier.numel() == 0:
                    non_outlier = channel.reshape(-1)
            else:
                non_outlier = channel.reshape(-1)

            quantized_channel, scale, zero_point = self._quantize_weight_channel(channel, non_outlier)
            if outlier_mask.any():
                quantized_channel = torch.where(
                    outlier_mask[channel_idx],
                    original_outliers[channel_idx],
                    quantized_channel,
                )
            quantized[channel_idx] = quantized_channel
            scales[channel_idx] = scale.reshape(1, 1, 1)
            zero_points[channel_idx] = zero_point.reshape(1, 1, 1)

        return quantized, scales, zero_points, outlier_mask, original_outliers

    def _quantize_weight_with_ocs(self, weight):
        per_input_mean = weight.abs().mean(dim=(0, 2, 3))
        duplicate_index = int(torch.argmax(per_input_mean).item())
        expanded = torch.cat([weight, weight[:, duplicate_index:duplicate_index + 1]], dim=1)
        expanded[:, duplicate_index, :, :] *= 0.5
        expanded[:, -1, :, :] *= 0.5

        original_observer = self.weight_observer
        if original_observer == "ocs":
            self.weight_observer = "minmax"
        quantized, scales, zero_points, outlier_mask, original_outliers = self._quantize_weight_tensor(expanded)
        self.weight_observer = original_observer

        self.ocs_duplicate_index.fill_(duplicate_index)
        return quantized, scales, zero_points, outlier_mask, original_outliers

    def quantize_weights(self):
        if self.skip_quantization:
            self.quantized_weight = self.original_weight.detach().clone()
            self.weights_quantized.fill_(True)
            self.activation_ready.fill_(True)
            return

        weight = self._weight_source()
        if self.weight_observer == "ocs":
            quantized, scales, zero_points, outlier_mask, original_outliers = self._quantize_weight_with_ocs(weight)
        else:
            self.ocs_duplicate_index.fill_(-1)
            quantized, scales, zero_points, outlier_mask, original_outliers = self._quantize_weight_tensor(weight)

        self.quantized_weight = quantized.detach().clone()
        self.weight_scale = scales.detach().clone()
        self.weight_zero_point = zero_points.detach().clone()
        self.outlier_mask = outlier_mask.detach().clone()
        self.original_outliers = original_outliers.detach().clone()
        qmin, qmax = get_qrange(self.nbits_w, symmetric=self.weight_symmetric)
        self.lb_w.data.copy_((qmin - self.weight_zero_point) * self.weight_scale)
        self.ub_w.data.copy_((qmax - self.weight_zero_point) * self.weight_scale)
        self.weights_quantized.fill_(True)

    def export_quant_state(self):
        return {
            "layer_name": self.layer_name,
            "layer_index": self.layer_index,
            "skip_quantization": self.skip_quantization,
            "lb_a": self.lb_a.detach().cpu(),
            "ub_a": self.ub_a.detach().cpu(),
            "bp_a": self.bp_a.detach().cpu(),
            "smooth_scale": self.smooth_scale.detach().cpu(),
            "activation_shift": self.activation_shift.detach().cpu(),
            "lb_w": self.lb_w.detach().cpu(),
            "ub_w": self.ub_w.detach().cpu(),
            "weight_scale": self.weight_scale.detach().cpu(),
            "weight_zero_point": self.weight_zero_point.detach().cpu(),
            "quantized_weight": self.quantized_weight.detach().cpu(),
            "outlier_mask": self.outlier_mask.detach().cpu(),
            "original_outliers": self.original_outliers.detach().cpu(),
            "weights_quantized": bool(self.weights_quantized.item()),
            "activation_ready": bool(self.activation_ready.item()),
            "ocs_duplicate_index": int(self.ocs_duplicate_index.item()),
        }

    def load_quant_state(self, state):
        self.layer_name = state.get("layer_name", self.layer_name)
        self.layer_index = state.get("layer_index", self.layer_index)
        self.skip_quantization = bool(state.get("skip_quantization", self.skip_quantization))
        self.lb_a.data.copy_(state["lb_a"].to(self.lb_a.device, self.lb_a.dtype))
        self.ub_a.data.copy_(state["ub_a"].to(self.ub_a.device, self.ub_a.dtype))
        if "bp_a" in state:
            self.bp_a.data.copy_(state["bp_a"].to(self.bp_a.device, self.bp_a.dtype))
        self.smooth_scale.data.copy_(state["smooth_scale"].to(self.smooth_scale.device, self.smooth_scale.dtype))
        if "activation_shift" in state:
            self.activation_shift.data.copy_(
                state["activation_shift"].to(self.activation_shift.device, self.activation_shift.dtype)
            )
        if "lb_w" in state:
            self.lb_w.data.copy_(state["lb_w"].to(self.lb_w.device, self.lb_w.dtype))
            self.ub_w.data.copy_(state["ub_w"].to(self.ub_w.device, self.ub_w.dtype))
        self.weight_scale = state["weight_scale"].to(self.weight_scale.device, self.weight_scale.dtype)
        self.weight_zero_point = state["weight_zero_point"].to(self.weight_zero_point.device, self.weight_zero_point.dtype)
        self.quantized_weight = state["quantized_weight"].to(self.weight.device, self.weight.dtype)
        self.outlier_mask = state["outlier_mask"].to(self.weight.device)
        self.original_outliers = state["original_outliers"].to(self.weight.device, self.weight.dtype)
        self.weights_quantized.fill_(bool(state.get("weights_quantized", True)))
        self.activation_ready.fill_(bool(state.get("activation_ready", True)))
        self.ocs_duplicate_index.fill_(int(state.get("ocs_duplicate_index", -1)))

    def _float_forward(self, x):
        return F.conv2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def _quantized_forward(self, x):
        x_q = self._apply_smooth_scale(x)
        if self.act_observer in {"oaptqsr", "wang2025", "wang2025_plq"}:
            upper = torch.maximum(self.lb_a.abs(), self.ub_a.abs())
            x_q = fake_quantize_piecewise_symmetric(
                x_q,
                breakpoint=self.bp_a,
                upper=upper,
                num_bits=self.nbits_a,
                dense_positive_bins=self.act_dense_positive_bins,
            )
        else:
            x_q = fake_quantize_from_bounds(
                x_q,
                self.lb_a,
                self.ub_a,
                num_bits=self.nbits_a,
                symmetric=self.act_symmetric,
            )
        if int(self.ocs_duplicate_index.item()) >= 0:
            idx = int(self.ocs_duplicate_index.item())
            x_q = torch.cat([x_q, x_q[:, idx:idx + 1, :, :]], dim=1)
            x_q[:, idx, :, :] *= 0.5
            x_q[:, -1, :, :] *= 0.5

        weight = self.quantized_weight
        if self.learn_weight_bounds:
            weight = fake_quantize_from_bounds(
                self._weight_source(),
                self.lb_w,
                self.ub_w,
                num_bits=self.nbits_w,
                symmetric=self.weight_symmetric,
            )
            if self.enable_weight_outlier_fp16 and self.outlier_mask.any():
                weight = torch.where(self.outlier_mask, self.original_outliers, weight)
        return F.conv2d(
            x_q,
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

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
