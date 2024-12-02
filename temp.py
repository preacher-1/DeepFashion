import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision import models
from torchvision.transforms import transforms

"""
网格表示+Transformer 编码器+解码器
使用ResNet18作为特征提取器，对原始图像进行特征提取，得到多尺度的特征图
对较大的特征图进行下采样后，将所有特征图进行网格划分，拼接后经过特征融合后送入Transformer编码器，得到全局特征表示（或不进行特征融合，直接经过位置编码输入Transformer编码器）
Transformer编码器输出的全局特征表示送入解码器，从<start>开始解码，生成图像描述文本。
"""


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class SE(nn.Module):
    def __init__(self, Cin, Cout):
        super(SE, self).__init__()
        num_hidden = max(Cout // 16, 4)
        self.se = nn.Sequential(nn.Linear(Cin, num_hidden), nn.ReLU(inplace=True),
                                nn.Linear(num_hidden, Cout), nn.Sigmoid())

    def forward(self, x):
        se = torch.mean(x, dim=[2, 3])
        se = se.view(se.size(0), -1)
        se = self.se(se)
        se = se.view(se.size(0), -1, 1, 1)
        return x * se


class Cell(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super(Cell, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, bias=False)
        self.bn_swish1 = nn.BatchNorm2d(in_channels * expansion)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels * expansion, in_channels * expansion, kernel_size=5, stride=stride, padding=2,
                      groups=in_channels * expansion),
            nn.BatchNorm2d(in_channels * expansion),
            nn.Conv2d(in_channels * expansion, in_channels * expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * expansion)
        )  # depthwise separable convolution
        self.bn_swish2 = nn.BatchNorm2d(in_channels * expansion)
        self.conv3 = nn.Conv2d(in_channels * expansion, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SE(out_channels, out_channels)

    def forward(self, x):
        skip = x
        x = self.bn1(x)  # C
        x = self.conv1(x)  # EC

        x = self.bn_swish1(x)
        x = F.silu(x)

        x = self.conv2(x)  # EC

        x = self.bn_swish2(x)
        x = F.silu(x)

        x = self.conv3(x)  # C
        x = self.bn2(x)
        x = self.se(x)

        return skip + x * 0.1  # skip connection and residual learning


class ConvTower(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvTower, self).__init__()
