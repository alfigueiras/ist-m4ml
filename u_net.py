import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


# Used in the encoder part of the UNet
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    # Uses positional embeddings
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


# Used in the decoder part of the UNet
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    # Uses both positional embeddings and skip connections
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


# Self Attention module used troughout the network, after downsample and upsample blocks
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class SimplifiedUNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.time_dim = time_dim
        self.start_dconv = DoubleConv(c_in, 64)
        self.downsample1 = DownsampleBlock(64, 128)
        self.sattention1 = SelfAttention(128)
        self.downsample2 = DownsampleBlock(128, 256)
        self.sattention2 = SelfAttention(256)
        self.downsample3 = DownsampleBlock(256, 256)
        self.sattention3 = SelfAttention(256)

        self.bottleneck_dconv1 = DoubleConv(256, 512)
        self.bottleneck_dconv2 = DoubleConv(512, 512)
        self.bottleneck_dconv3 = DoubleConv(512, 256)

        self.upsample1 = UpsampleBlock(512, 128)
        self.sattention4 = SelfAttention(128)
        self.upsample2 = UpsampleBlock(256, 64)
        self.sattention5 = SelfAttention(64)
        self.upsample3 = UpsampleBlock(128, 64)
        self.sattention6 = SelfAttention(64)
        self.end_conv = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        """
        Adapted positional encoding implementation from PyTorch docs
        """
        div_term = torch.exp(torch.arange(0, channels, 2) * (-math.log(10000.0) / channels)).to(self.device)
        pos_enc = torch.zeros(t.shape[0], channels, device=self.device)
        pos_enc[:, 0::2] = torch.sin(t * div_term)
        pos_enc[:, 1::2] = torch.cos(t * div_term)

        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)[:x.size(0)]

        # Encoder
        x1 = self.start_dconv(x)
        x2 = self.downsample1(x1, t)
        x2 = self.sattention1(x2)
        x3 = self.downsample2(x2, t)
        x3 = self.sattention2(x3)
        x4 = self.downsample3(x3, t)
        x4 = self.sattention3(x4)

        # Bottleneck
        x4 = self.bottleneck_dconv1(x4)
        x4 = self.bottleneck_dconv2(x4)
        x4 = self.bottleneck_dconv3(x4)

        # Decoder
        x = self.upsample1(x4, x3, t)
        x = self.sattention4(x)
        x = self.upsample2(x, x2, t)
        x = self.sattention5(x)
        x = self.upsample3(x, x1, t)
        x = self.sattention6(x)
        output = self.end_conv(x)
        return output