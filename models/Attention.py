import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        # self.se = nn.Sequential(
        #     nn.Conv2d(20, 10, 1, bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(10, 1, 1, bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(1, 10, 1, bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(10, 20, 1, bias=False),
        #     )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        final = max_out + avg_out
        # final[torch.abs(final) < 0.1] = 0
        output = self.sigmoid(final)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)  # 这个是怎么卷的，2通道变成1通道？？？
        output = self.sigmoid(output)
        return output


