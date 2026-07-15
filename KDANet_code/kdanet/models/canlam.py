import math

import torch
from torch import nn


class ChannelAttention(nn.Module):
    """Channel attention branch used in CANLAM."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(self.avg_pool(x))
        return x * weights


class NonLocalAttention(nn.Module):
    """Non-local attention branch for global contextual dependency modeling."""

    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.key = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w

        query = self.query(x).view(b, c, n).transpose(1, 2)
        key = self.key(x).view(b, c, n)
        value = self.value(x).view(b, c, n).transpose(1, 2)

        attention = torch.bmm(query, key) / math.sqrt(float(c))
        attention = torch.softmax(attention, dim=-1)
        context = torch.bmm(attention, value).transpose(1, 2).contiguous()
        context = context.view(b, c, h, w)

        return self.out(context) + x


class CANLAM(nn.Module):
    """Channel-aware non-local attention module.

    The module first recalibrates channels and then models global spatial
    dependencies in the refined feature map.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.non_local_attention = NonLocalAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.non_local_attention(x)
        return x
