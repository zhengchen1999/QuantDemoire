"""ESDNet implementation for image demoiréing."""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class _EncoderFeatures:
    """Owned single-use container, so a module argument does not retain old features."""
    def __init__(self, features):
        self.parts = list(features)
        if len(self.parts) != 3:
            raise ValueError("expected three encoder features")

    def pop(self):
        return self.parts.pop()


class _LevelInput:
    """Owned level input, released after its residual dense block finishes."""
    def __init__(self, tensor):
        self.tensor = tensor

    def pop(self):
        tensor = self.tensor
        if tensor is None:
            raise ValueError("consumed level input")
        self.tensor = None
        return tensor


class ESDNet(nn.Module):
    def __init__(
        self,
        en_feature_num: int,
        en_inter_num: int,
        de_feature_num: int,
        de_inter_num: int,
        sam_number: int = 1,
    ) -> None:
        super().__init__()
        self.release_features = os.environ.get("QUANTDEMOIRE_RELEASE_FEATURES", "0") == "1"
        self.encoder = Encoder(en_feature_num, en_inter_num, sam_number)
        self.decoder = Decoder(
            en_feature_num, de_feature_num, de_inter_num, sam_number
        )

    def forward(self, tensor: torch.Tensor):
        if self.release_features:
            return self.decoder(_EncoderFeatures(self.encoder(tensor)))
        feature_1, feature_2, feature_3 = self.encoder(tensor)
        return self.decoder(feature_1, feature_2, feature_3)


class Decoder(nn.Module):
    def __init__(self, en_num: int, feature_num: int, inter_num: int, sam_number: int):
        super().__init__()
        self.preconv_3 = ConvReLU(4 * en_num, feature_num, 3, padding=1)
        self.decoder_3 = DecoderLevel(feature_num, inter_num, sam_number)
        self.preconv_2 = ConvReLU(2 * en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_2 = DecoderLevel(feature_num, inter_num, sam_number)
        self.preconv_1 = ConvReLU(en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_1 = DecoderLevel(feature_num, inter_num, sam_number)

    def forward(self, feature_1, feature_2=None, feature_3=None):
        if isinstance(feature_1, _EncoderFeatures):
            if feature_2 is not None or feature_3 is not None or len(feature_1.parts) != 3:
                raise ValueError("invalid or reused encoder feature queue")
            return self._consume_features(feature_1)
        if feature_2 is None or feature_3 is None:
            raise TypeError("Decoder expects three tensors or an owned feature queue")
        if getattr(self, 'lazy_concat', False):
            from .deploy import ChannelConcat
            concatenate = lambda parts: ChannelConcat(parts)
        else:
            concatenate = lambda parts: torch.cat(parts, dim=1)
        level_3 = self.preconv_3(feature_3)
        output_3, decoded_3 = self.decoder_3(level_3)
        level_2 = self.preconv_2(concatenate([feature_2, decoded_3]))
        output_2, decoded_2 = self.decoder_2(level_2)
        level_1 = self.preconv_1(concatenate([feature_1, decoded_2]))
        output_1 = self.decoder_1(level_1, return_features=False)
        return output_1, output_2, output_3

    def _consume_features(self, features):
        # A three-tensor argument tuple would retain the encoder outputs in
        # Module.__call__ even after deleting this forward's own references.
        if getattr(self, 'lazy_concat', False):
            from .deploy import ChannelConcat
            concatenate = lambda parts: ChannelConcat(parts)
        else:
            concatenate = lambda parts: torch.cat(parts, dim=1)
        last = features.pop()
        level_3 = self.preconv_3(last)
        del last
        level_3 = _LevelInput(level_3)
        output_3, decoded_3 = self.decoder_3(level_3)
        del level_3
        middle = features.pop()
        combined = concatenate([middle, decoded_3])
        level_2 = self.preconv_2(combined)
        del combined, middle, decoded_3
        level_2 = _LevelInput(level_2)
        output_2, decoded_2 = self.decoder_2(level_2)
        del level_2
        first = features.pop()
        combined = concatenate([first, decoded_2])
        level_1 = self.preconv_1(combined)
        del combined, first, decoded_2
        level_1 = _LevelInput(level_1)
        output_1 = self.decoder_1(level_1, return_features=False)
        assert not features.parts
        return output_1, output_2, output_3


class Encoder(nn.Module):
    def __init__(self, feature_num: int, inter_num: int, sam_number: int):
        super().__init__()
        self.release_features = os.environ.get("QUANTDEMOIRE_RELEASE_FEATURES", "0") == "1"
        self.conv_first = nn.Sequential(
            nn.Conv2d(12, feature_num, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.encoder_1 = EncoderLevel(feature_num, inter_num, 1, sam_number)
        self.encoder_2 = EncoderLevel(2 * feature_num, inter_num, 2, sam_number)
        self.encoder_3 = EncoderLevel(4 * feature_num, inter_num, 3, sam_number)

    def forward(self, tensor):
        tensor = self.conv_first(F.pixel_unshuffle(tensor, 2))
        if self.release_features:
            tensor = _LevelInput(tensor)
            output_1, tensor = self.encoder_1(tensor)
            tensor = _LevelInput(tensor)
            output_2, tensor = self.encoder_2(tensor)
            tensor = _LevelInput(tensor)
            output_3 = self.encoder_3(tensor)
            return output_1, output_2, output_3
        output_1, down_1 = self.encoder_1(tensor)
        output_2, down_2 = self.encoder_2(down_1)
        output_3 = self.encoder_3(down_2)
        return output_1, output_2, output_3


class EncoderLevel(nn.Module):
    def __init__(
        self, feature_num: int, inter_num: int, level: int, sam_number: int
    ) -> None:
        super().__init__()
        self.rdb = ResidualDenseBlock(feature_num, (1, 2, 1), inter_num)
        self.sam_blocks = nn.ModuleList(
            ScaleAwareModule(feature_num, (1, 2, 3, 2, 1), inter_num)
            for _ in range(sam_number)
        )
        self.level = level
        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, tensor):
        if isinstance(tensor, _LevelInput):
            tensor = tensor.pop()
        output = self.rdb(tensor)
        del tensor
        for block in self.sam_blocks:
            output = block(output)
        if self.level < 3:
            return output, self.down(output)
        return output


class DecoderLevel(nn.Module):
    def __init__(self, feature_num: int, inter_num: int, sam_number: int):
        super().__init__()
        self.rdb = ResidualDenseBlock(feature_num, (1, 2, 1), inter_num)
        self.sam_blocks = nn.ModuleList(
            ScaleAwareModule(feature_num, (1, 2, 3, 2, 1), inter_num)
            for _ in range(sam_number)
        )
        self.conv = Conv(feature_num, 12, kernel_size=3, padding=1)

    def forward(self, tensor, return_features=True):
        if isinstance(tensor, _LevelInput):
            tensor = tensor.pop()
        tensor = self.rdb(tensor)
        for block in self.sam_blocks:
            tensor = block(tensor)
        output = F.pixel_shuffle(self.conv(tensor), 2)
        if return_features:
            return output, F.interpolate(tensor, scale_factor=2, mode="bilinear")
        return output


class DenseBlock(nn.Module):
    def __init__(self, in_channel: int, dilations: tuple[int, ...], inter_num: int):
        super().__init__()
        channels = in_channel
        self.conv_layers = nn.ModuleList()
        for dilation in dilations:
            self.conv_layers.append(
                ConvReLU(
                    channels,
                    inter_num,
                    kernel_size=3,
                    dilation=dilation,
                    padding=dilation,
                )
            )
            channels += inter_num
        self.conv_post = Conv(channels, in_channel, kernel_size=1)
        self.fuse_residual = os.environ.get('QUANTDEMOIRE_FUSE_RESIDUAL','0') == '1'

    def forward(self, tensor, residual=None):
        output = tensor
        pending_second = None
        for i, layer in enumerate(self.conv_layers):
            if getattr(self, 'lazy_concat', False):
                from .deploy import ChannelConcat
                producer = layer.conv[0]
                consumer = self.conv_layers[i+1].conv[0] if i+1 < len(self.conv_layers) else self.conv_post.conv
                if getattr(self, 'fuse_next_quant', False) and producer.can_quantize_for(consumer):
                    future = None
                    if i+2 <= len(self.conv_layers):
                        candidate = self.conv_layers[i+2].conv[0] if i+2 < len(self.conv_layers) else self.conv_post.conv
                        n,_,h,w=output.shape
                        if (producer.can_quantize_second_for(consumer,candidate)
                            and n*producer.sparse_channels*h*w<2147483647):
                            future=candidate
                    # Earlier dense features still have uncached consumers.
                    # The final feature needs one cache; the penultimate needs two.
                    terminal=(i==len(self.conv_layers)-1 or
                              (i==len(self.conv_layers)-2 and future is not None))
                    omit=(producer.omit_cached_features>0 and terminal
                          and (producer.joint_warps>0 or producer.omit_cached_features==2)
                          and consumer.prepare_impl=='quad_swizzle'
                          and producer.can_omit_cached_feature(output))
                    result = producer(output, next_quantizer=consumer, future_quantizer=future, omit_feature=omit)
                    feature, packed, planar = result[:3]
                    output = ChannelConcat([feature, output], prepared=(consumer,packed,planar,feature.shape[1]),second_cached=pending_second)
                    pending_second = result[3] if future is not None else None
                    del result, packed, planar
                else:
                    if pending_second is not None:raise RuntimeError('second cache lost its first-prefix producer')
                    output = ChannelConcat([layer(output), output])
            else:
                output = torch.cat([layer(output), output], dim=1)
        if pending_second is not None:raise RuntimeError('unconsumed second cache')
        if residual is not None:
            return self.conv_post.conv(output,residual=residual)
        return self.conv_post(output)


class ScaleAwareModule(nn.Module):
    def __init__(self, in_channel: int, dilations: tuple[int, ...], inter_num: int):
        super().__init__()
        self.basic_block = DenseBlock(in_channel, dilations, inter_num)
        self.basic_block_2 = DenseBlock(in_channel, dilations, inter_num)
        self.basic_block_4 = DenseBlock(in_channel, dilations, inter_num)
        self.fusion = ChannelScaleAttention(3 * in_channel)

    def forward(self, tensor):
        full = self.basic_block(tensor)
        half = F.interpolate(
            self.basic_block_2(F.interpolate(tensor, scale_factor=0.5, mode="bilinear")),
            scale_factor=2,
            mode="bilinear",
        )
        quarter = F.interpolate(
            self.basic_block_4(F.interpolate(tensor, scale_factor=0.25, mode="bilinear")),
            scale_factor=4,
            mode="bilinear",
        )
        return self.fusion(full, half, quarter, residual=tensor)


class ChannelScaleAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        hidden = channels // ratio
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1 = nn.Conv2d(channels, hidden, 1)
        self.compress2 = nn.Conv2d(hidden, hidden, 1)
        self.excitation = nn.Conv2d(hidden, channels, 1)
        self.fuse_attention = os.environ.get('QUANTDEMOIRE_FUSE_ATTENTION','0') == '1'

    def forward(self, full, half, quarter, residual=None):
        weights = torch.cat(
            [self.squeeze(full), self.squeeze(half), self.squeeze(quarter)], dim=1
        )
        weights = F.relu(self.compress1(weights))
        weights = F.relu(self.compress2(weights))
        weights = torch.sigmoid(self.excitation(weights))
        if self.fuse_attention and not torch.is_grad_enabled() and residual is not None and full.is_cuda and full.dtype == torch.float16:
            from kernel.ops import extension
            return extension().attention_residual(full,half,quarter,weights,residual)
        full_weight, half_weight, quarter_weight = torch.chunk(weights,3,dim=1)
        result = full * full_weight + half * half_weight + quarter * quarter_weight
        return result if residual is None else residual + result


class ResidualDenseBlock(DenseBlock):
    def forward(self, tensor):
        conv=self.conv_post.conv
        if self.fuse_residual and getattr(conv,'supports_residual',lambda:False)():
            return super().forward(tensor,residual=tensor)
        return super().forward(tensor) + tensor


class Conv(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        dilation: int = 1,
        padding: int = 0,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, tensor):
        return self.conv(tensor)


class ConvReLU(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        dilation: int = 1,
        padding: int = 0,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, tensor):
        return self.conv(tensor)


# Keep the original checkpoint-facing name available.
my_model = ESDNet
