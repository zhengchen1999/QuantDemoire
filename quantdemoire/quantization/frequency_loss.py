"""Frequency-aware calibration losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .observers import build_radial_mask


def _cfg(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def gaussian_blur(image, radius):
    kernel = image.new_tensor(
        [
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625],
        ]
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(image.size(1), 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(image, kernel, groups=image.size(1), dilation=radius)


def gaussian_lowpass(image, levels=3, first_dilation=1):
    if levels < 0 or first_dilation < 1:
        raise ValueError('invalid Gaussian extraction configuration')
    low = image
    for level in range(levels):
        low = gaussian_blur(low, radius=first_dilation * 2 ** level)
    return low


def gaussian_lowpass_loss(pred, target, levels=3, aux_full_l1_weight=0.0):
    pred_low = gaussian_lowpass(pred, levels=levels)
    target_low = gaussian_lowpass(target, levels=levels)
    loss = F.l1_loss(pred_low, target_low)
    if aux_full_l1_weight > 0:
        loss = loss + aux_full_l1_weight * F.l1_loss(pred, target)
    return loss


def _pad_even(image):
    pad_height = image.size(-2) % 2
    pad_width = image.size(-1) % 2
    if pad_height or pad_width:
        image = F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")
    return image


def haar_decompose(image):
    """One-level orthonormal 2-D Haar decomposition."""

    image = _pad_even(image)
    x00 = image[..., 0::2, 0::2]
    x01 = image[..., 0::2, 1::2]
    x10 = image[..., 1::2, 0::2]
    x11 = image[..., 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (-x00 - x01 + x10 + x11) * 0.5
    hl = (-x00 + x01 - x10 + x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, (lh, hl, hh)


def haar_wavelet_loss(
    pred,
    target,
    levels=3,
    low_weight=1.0,
    detail_weight=1.0,
    aux_full_l1_weight=0.0,
):
    """Multi-level Haar loss over the approximation and detail sub-bands."""

    pred_low = pred
    target_low = target
    loss = pred.new_tensor(0.0)
    for _ in range(levels):
        pred_low, pred_details = haar_decompose(pred_low)
        target_low, target_details = haar_decompose(target_low)
        if detail_weight > 0:
            loss = loss + detail_weight * sum(
                F.l1_loss(pred_band, target_band)
                for pred_band, target_band in zip(pred_details, target_details)
            ) / len(pred_details)
    if low_weight > 0:
        loss = loss + low_weight * F.l1_loss(pred_low, target_low)
    if aux_full_l1_weight > 0:
        loss = loss + aux_full_l1_weight * F.l1_loss(pred, target)
    return loss


class VGGFeatureLoss(nn.Module):
    """Frozen ImageNet VGG16 feature loss used by paper-aligned FAC runs."""

    def __init__(self, feature_indices=(15,), preprocess='legacy_clamped'):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        self.features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        self.feature_indices = {int(index) for index in feature_indices}
        if not self.feature_indices or min(self.feature_indices)<0 or max(self.feature_indices)>15:
            raise ValueError('VGG feature indices must be within 0–15')
        if preprocess not in ('legacy_clamped','esdnet'):
            raise ValueError('unsupported VGG preprocessing')
        self.preprocess = preprocess
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(False)
        self.features.eval()
        return self

    def forward(self, pred, target):
        if self.preprocess == 'esdnet':
            # The cited ESDNet loss normalizes before resize and does not clamp.
            pred = (pred-self.mean)/self.std
            target = (target-self.mean)/self.std
        pred = F.interpolate(pred, size=(224, 224), mode="bilinear", align_corners=False)
        target = F.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)
        if self.preprocess == 'legacy_clamped':
            pred = (pred.clamp(0.0, 1.0) - self.mean) / self.std
            target = (target.clamp(0.0, 1.0) - self.mean) / self.std
        loss = pred.new_tensor(0.0)
        pred_feature = pred
        target_feature = target
        for index, layer in enumerate(self.features):
            pred_feature = layer(pred_feature)
            with torch.no_grad():
                target_feature = layer(target_feature)
            if index in self.feature_indices:
                loss = loss + F.l1_loss(pred_feature, target_feature)
        return loss


def fft_amplitude(image, use_log_mag=True):
    spec = torch.fft.rfft2(image, norm="ortho")
    amplitude = spec.abs()
    if use_log_mag:
        amplitude = torch.log1p(amplitude)
    return amplitude


def fft_band_loss(
    pred,
    target,
    low_ratio=0.15,
    high_ratio=0.45,
    low_weight=1.0,
    mid_weight=1.0,
    high_weight=0.0,
    use_log_mag=True,
    aux_full_l1_weight=0.0,
):
    pred_fft = fft_amplitude(pred, use_log_mag=use_log_mag)
    target_fft = fft_amplitude(target, use_log_mag=use_log_mag)
    low_mask, mid_mask, high_mask = build_radial_mask(
        height=pred.size(-2),
        width=pred.size(-1),
        low_ratio=low_ratio,
        high_ratio=high_ratio,
        device=pred.device,
    )
    low_mask = low_mask.view(1, 1, *low_mask.shape)
    mid_mask = mid_mask.view(1, 1, *mid_mask.shape)
    high_mask = high_mask.view(1, 1, *high_mask.shape)

    loss = pred.new_tensor(0.0)
    if low_weight > 0:
        loss = loss + low_weight * F.l1_loss(pred_fft * low_mask, target_fft * low_mask)
    if mid_weight > 0:
        loss = loss + mid_weight * F.l1_loss(pred_fft * mid_mask, target_fft * mid_mask)
    if high_weight > 0:
        loss = loss + high_weight * F.l1_loss(pred_fft * high_mask, target_fft * high_mask)
    if aux_full_l1_weight > 0:
        loss = loss + aux_full_l1_weight * F.l1_loss(pred, target)
    return loss


class FrequencyCalibrationLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.loss_type = str(_cfg(config, "loss_type", "gaussian")).lower()
        self.gaussian_levels = int(_cfg(config, "gaussian_levels", 3))
        self.gaussian_first_dilation = int(_cfg(config, 'gaussian_first_dilation', 1))
        self.aux_full_l1_weight = float(_cfg(config, "aux_full_l1_weight", 0.1))
        self.perceptual_weight = float(_cfg(config, "perceptual_weight", 0.0))
        self.perceptual_feature_indices = _cfg(config, "perceptual_feature_indices", [15])
        self.fft_low_ratio = float(_cfg(config, "fft_low_ratio", 0.15))
        self.fft_high_ratio = float(_cfg(config, "fft_high_ratio", 0.45))
        self.fft_low_weight = float(_cfg(config, "fft_low_weight", 1.0))
        self.fft_mid_weight = float(_cfg(config, "fft_mid_weight", 1.0))
        self.fft_high_weight = float(_cfg(config, "fft_high_weight", 0.0))
        self.fft_use_log_mag = bool(_cfg(config, "fft_use_log_mag", True))
        self.wavelet_levels = int(_cfg(config, "wavelet_levels", 3))
        self.wavelet_low_weight = float(_cfg(config, "wavelet_low_weight", 1.0))
        self.wavelet_detail_weight = float(_cfg(config, "wavelet_detail_weight", 1.0))
        self.perceptual = (
            VGGFeatureLoss(feature_indices=self.perceptual_feature_indices,
                           preprocess=_cfg(config,'perceptual_preprocess','legacy_clamped'))
            if self.perceptual_weight > 0
            else None
        )

    def _single_scale_loss(self, pred, target):
        if self.loss_type == "fft":
            return fft_band_loss(
                pred,
                target,
                low_ratio=self.fft_low_ratio,
                high_ratio=self.fft_high_ratio,
                low_weight=self.fft_low_weight,
                mid_weight=self.fft_mid_weight,
                high_weight=self.fft_high_weight,
                use_log_mag=self.fft_use_log_mag,
                aux_full_l1_weight=self.aux_full_l1_weight,
            )
        if self.loss_type == "wavelet":
            return haar_wavelet_loss(
                pred,
                target,
                levels=self.wavelet_levels,
                low_weight=self.wavelet_low_weight,
                detail_weight=self.wavelet_detail_weight,
                aux_full_l1_weight=self.aux_full_l1_weight,
            )
        pred_low = gaussian_lowpass(pred, levels=self.gaussian_levels, first_dilation=self.gaussian_first_dilation)
        target_low = gaussian_lowpass(target, levels=self.gaussian_levels, first_dilation=self.gaussian_first_dilation)
        loss = F.l1_loss(pred_low, target_low)
        if self.perceptual is not None:
            loss = loss + self.perceptual_weight * self.perceptual(pred_low, target_low)
        if self.aux_full_l1_weight > 0:
            loss = loss + self.aux_full_l1_weight * F.l1_loss(pred, target)
        return loss

    def forward(self, out_1, out_2, out_3, target):
        target_2 = F.interpolate(target, scale_factor=0.5, mode="bilinear", align_corners=False)
        target_3 = F.interpolate(target, scale_factor=0.25, mode="bilinear", align_corners=False)
        return (
            self._single_scale_loss(out_1, target)
            + self._single_scale_loss(out_2, target_2)
            + self._single_scale_loss(out_3, target_3)
        )


class ReconstructionCalibrationLoss(nn.Module):
    def __init__(self, l1_weight=1.0):
        super().__init__()
        self.l1_weight = float(l1_weight)

    def forward(self, out_1, out_2, out_3, target):
        target_2 = F.interpolate(target, scale_factor=0.5, mode="bilinear", align_corners=False)
        target_3 = F.interpolate(target, scale_factor=0.25, mode="bilinear", align_corners=False)
        return self.l1_weight * (
            F.l1_loss(out_1, target)
            + F.l1_loss(out_2, target_2)
            + F.l1_loss(out_3, target_3)
        )


def build_frequency_loss(config=None):
    return FrequencyCalibrationLoss(config=config)


def build_reconstruction_loss(config=None):
    l1_weight = _cfg(config, "reconstruction_l1_weight", 1.0)
    return ReconstructionCalibrationLoss(l1_weight=l1_weight)
