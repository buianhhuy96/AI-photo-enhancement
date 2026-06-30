"""BiSeNet face parsing model.

Source: https://github.com/yakhyo/face-parsing (MIT License)
Author: Yakhyokhuja Valikhujaev
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from backend.models.resnet import resnet18, resnet34
from typing import Union, Optional, Tuple


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, groups=1, dilation=1, inplace=True, bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.norm(self.conv(x)))


class BiSeNetOutput(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.conv_block = ConvBNReLU(in_channels, mid_channels, kernel_size=3, stride=1)
        self.conv = nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.conv_block(x))


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=1)
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        feat = self.conv_block(x)
        feat_shape = [int(t) for t in feat.size()[2:]]
        pool = F.avg_pool2d(feat, feat_shape)
        attention = self.attention(pool)
        return torch.mul(feat, attention)


class ContextPath(nn.Module):
    def __init__(self, backbone_name='resnet18'):
        super().__init__()
        if backbone_name == 'resnet18':
            self.backbone = resnet18()
        elif backbone_name == 'resnet34':
            self.backbone = resnet34()
        else:
            raise ValueError(f'Available backbone modules: resnet18, resnet34')
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, kernel_size=3, stride=1)
        self.conv_head16 = ConvBNReLU(128, 128, kernel_size=3, stride=1)
        self.conv_avg = ConvBNReLU(512, 128, kernel_size=1, stride=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        feat8, feat16, feat32 = self.backbone(x)
        h8, w8 = feat8.size()[2:]
        h16, w16 = feat16.size()[2:]
        h32, w32 = feat32.size()[2:]

        feat32_shape = [int(t) for t in feat32.size()[2:]]
        avg = F.avg_pool2d(feat32, feat32_shape)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (h32, w32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (h16, w16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (h8, w8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp: Tensor, fcp: Tensor) -> Tensor:
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.conv_block(fcat)
        feat_shape = [int(t) for t in feat.size()[2:]]
        attention = F.avg_pool2d(feat, feat_shape)
        attention = self.relu(self.conv1(attention))
        attention = self.sigmoid(self.conv2(attention))
        return torch.mul(feat, attention) + feat


class BiSeNet(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet18'):
        super().__init__()
        self.fpn = ContextPath(backbone_name=backbone_name)
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, num_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, num_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, num_classes)

    def forward(self, x):
        h, w = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.fpn(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)
        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)
        feat_out = F.interpolate(feat_out, (h, w), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (h, w), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (h, w), mode='bilinear', align_corners=True)
        return feat_out, feat_out16, feat_out32
