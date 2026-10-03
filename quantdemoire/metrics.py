"""Quality metrics for QuantDemoire evaluation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, channels: int, device, dtype):
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates -= window_size // 2
    kernel_1d = torch.exp(-(coordinates**2) / (2 * 1.5**2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def structural_similarity(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    _, channels, height, width = first.shape
    window_size = min(11, height, width)
    window = _gaussian_window(window_size, channels, first.device, first.dtype)
    mean_first = F.conv2d(first, window, groups=channels)
    mean_second = F.conv2d(second, window, groups=channels)
    mean_first_sq = mean_first.square()
    mean_second_sq = mean_second.square()
    mean_product = mean_first * mean_second
    variance_first = F.conv2d(first.square(), window, groups=channels) - mean_first_sq
    variance_second = F.conv2d(second.square(), window, groups=channels) - mean_second_sq
    covariance = F.conv2d(first * second, window, groups=channels) - mean_product
    c1 = 0.01**2
    c2 = 0.03**2
    return (
        ((2 * mean_product + c1) * (2 * covariance + c2))
        / ((mean_first_sq + mean_second_sq + c1) * (variance_first + variance_second + c2))
    ).mean()


class QualityMetrics(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError(
                "quality evaluation requires lpips; install requirements.txt"
            ) from error
        self.lpips = lpips.LPIPS(net="alex").to(device).eval()

    @torch.inference_mode()
    def compute(self, prediction: torch.Tensor, target: torch.Tensor):
        prediction = prediction.clamp(0, 1)
        target = target.clamp(0, 1)
        mse = F.mse_loss(prediction, target)
        psnr = -10 * torch.log10(mse.clamp_min(torch.finfo(mse.dtype).tiny))
        ssim = structural_similarity(prediction, target)
        perceptual = self.lpips(prediction, target, normalize=True).item()
        return perceptual, psnr.item(), ssim.item()
