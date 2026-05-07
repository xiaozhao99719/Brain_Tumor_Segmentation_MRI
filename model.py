"""
model.py — BrainMRI 分割模型定义
==================================
包含三种 3D 医学图像分割模型：
  1. nnU-Net   — 动态 U-Net (简化版, 无 nnU-Net 框架依赖)
  2. Attention U-Net — 带注意力门控的 U-Net
  3. TransUNet — ViT 编码器 + CNN 解码器混合架构
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  通用组件
# ═══════════════════════════════════════════════════════════════


class ConvBlock(nn.Module):
    """Conv3d → InstanceNorm3d → LeakyReLU (×2), nnU-Net 风格。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """下采样: MaxPool3d + ConvBlock。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """上采样: 转置卷积 + 拼接 skip + ConvBlock。"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 处理尺寸不匹配 (奇数尺寸导致上采样后多1像素)
        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2,
                       diff_d // 2, diff_d - diff_d // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════
#  1. nnU-Net (简化动态 U-Net)
# ═══════════════════════════════════════════════════════════════


class nnUNet(nn.Module):
    """
    nnU-Net 简化实现。

    与原始 nnU-Net 相比主要简化:
      - 去掉残差连接 / 深度监督等高级特性
      - 使用固定的 5 级编码器-解码器结构
      - 保留 InstanceNorm + LeakyReLU + Conv3d 的核心风格

    Parameters
    ----------
    in_channels : 输入通道数 (MRI 模态数, 默认 4)
    num_classes : 输出类别数 (含背景)
    base_filters : 第一层滤波器数 (后续层 ×2 递增)
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_filters: int = 32,
    ):
        super().__init__()
        f = base_filters  # 32

        # 编码器
        self.enc1 = ConvBlock(in_channels, f)          # → 32
        self.enc2 = Down(f, f * 2)                     # → 64
        self.enc3 = Down(f * 2, f * 4)                 # → 128
        self.enc4 = Down(f * 4, f * 8)                 # → 256
        self.bottleneck = Down(f * 8, f * 16)          # → 512

        # 解码器
        self.up4 = Up(f * 16, f * 8, f * 8)            # 512+256 → 256
        self.up3 = Up(f * 8, f * 4, f * 4)             # 256+128 → 128
        self.up2 = Up(f * 4, f * 2, f * 2)             # 128+64  → 64
        self.up1 = Up(f * 2, f, f)                     # 64+32   → 32

        # 输出头
        self.seg_head = nn.Conv3d(f, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)

        # 解码
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.seg_head(d1)


# ═══════════════════════════════════════════════════════════════
#  2. Attention U-Net
# ═══════════════════════════════════════════════════════════════


class AttentionGate(nn.Module):
    """
    3D 注意力门控。
    对 skip connection 进行软门控，抑制无关特征响应。
    """

    def __init__(self, f_g: int, f_x: int, f_int: int):
        """
        Parameters
        ----------
        f_g : 门控信号 (来自解码器低层) 通道数
        f_x : skip 特征通道数
        f_int : 中间压缩通道数
        """
        super().__init__()
        self.w_g = nn.Conv3d(f_g, f_int, kernel_size=1, bias=False)
        self.w_x = nn.Conv3d(f_x, f_int, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv3d(f_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        # g1 可能空间尺寸更小 → 上采样到 x 尺寸
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="trilinear", align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """
    Attention U-Net (3D)。

    在标准 U-Net 解码器上每级加入 Attention Gate，
    自动学习聚焦于目标区域。

    Parameters
    ----------
    in_channels : 输入通道数
    num_classes : 输出类别数
    base_filters : 基础滤波器数
    attention_gate_channels : 注意力门控中间通道数
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_filters: int = 32,
        attention_gate_channels: int = 128,
    ):
        super().__init__()
        f = base_filters
        agc = attention_gate_channels

        # 编码器 (与 nnUNet 相同结构)
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = Down(f, f * 2)
        self.enc3 = Down(f * 2, f * 4)
        self.enc4 = Down(f * 4, f * 8)
        self.bottleneck = Down(f * 8, f * 16)

        # 注意力门控
        self.ag4 = AttentionGate(f * 16, f * 8, agc)
        self.ag3 = AttentionGate(f * 8, f * 4, agc)
        self.ag2 = AttentionGate(f * 4, f * 2, agc)
        self.ag1 = AttentionGate(f * 2, f, agc)

        # 解码器
        self.up4 = Up(f * 16, f * 8, f * 8)
        self.up3 = Up(f * 8, f * 4, f * 4)
        self.up2 = Up(f * 4, f * 2, f * 2)
        self.up1 = Up(f * 2, f, f)

        self.seg_head = nn.Conv3d(f, num_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)

        # 门控 skip
        g4 = self.ag4(b, e4)
        d4 = self.up4(b, g4)

        g3 = self.ag3(d4, e3)
        d3 = self.up3(d4, g3)

        g2 = self.ag2(d3, e2)
        d2 = self.up2(d3, g2)

        g1 = self.ag1(d2, e1)
        d1 = self.up1(d2, g1)

        return self.seg_head(d1)


# ═══════════════════════════════════════════════════════════════
#  3. TransUNet
# ═══════════════════════════════════════════════════════════════


class PatchEmbedding3D(nn.Module):
    """将 3D 体积切分为 patch 并线性映射为 token 序列。"""

    def __init__(self, in_channels: int, img_size: int, patch_size: int,
                 hidden_size: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 3
        self.proj = nn.Conv3d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        assert D == H == W == self.img_size, (
            f"输入尺寸 {D}x{H}x{W} 不等于 img_size={self.img_size}"
        )
        x = self.proj(x)                            # (B, hidden, D/p, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)            # (B, N, hidden)
        return x


class TransformerEncoderBlock(nn.Module):
    """标准 ViT Transformer Encoder Block。"""

    def __init__(self, hidden_size: int, num_heads: int, mlp_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """ViT 编码器: PatchEmbed + [CLS] + Positional Encoding + N 层 Transformer。"""

    def __init__(self, in_channels: int, img_size: int, patch_size: int,
                 hidden_size: int, num_heads: int, num_layers: int,
                 mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding3D(in_channels, img_size, patch_size,
                                            hidden_size)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, hidden_size)
        )
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(hidden_size, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)                     # (B, N, hidden)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)       # (B, N+1, hidden)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x                                    # (B, N+1, hidden)


class DecoderBlock(nn.Module):
    """TransUNet 解码器模块: 上采样 + 拼接 skip + 卷积。"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 尺寸对齐
        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2,
                       diff_d // 2, diff_d - diff_d // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TransUNet(nn.Module):
    """
    TransUNet (3D)。

    编码器前 3 级用 CNN 提取多尺度特征，
    瓶颈层用 ViT 建模全局依赖，
    解码器逐级上采样恢复分辨率。

    Parameters
    ----------
    in_channels : 输入通道数
    num_classes : 输出类别数
    img_size : ViT 输入体积尺寸 (需能被 patch_size 整除)
    patch_size : ViT patch 尺寸
    hidden_size : ViT 隐层维度
    num_heads : ViT 注意力头数
    num_layers : ViT Transformer 层数
    mlp_dim : ViT MLP 中间维度
    dropout : ViT dropout 率
    base_filters : CNN 编码器基础通道数
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        img_size: int = 128,
        patch_size: int = 16,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        base_filters: int = 32,
    ):
        super().__init__()
        f = base_filters
        self.img_size = img_size

        # ── CNN 编码器 (3 级) ──
        self.enc1 = ConvBlock(in_channels, f)       # 32
        self.pool1 = nn.MaxPool3d(2, 2)
        self.enc2 = ConvBlock(f, f * 2)              # 64
        self.pool2 = nn.MaxPool3d(2, 2)
        self.enc3 = ConvBlock(f * 2, f * 4)          # 128
        self.pool3 = nn.MaxPool3d(2, 2)

        # ── ViT 瓶颈 ──
        # 经过 3 次 pool 后, 尺寸 = img_size // 8
        vit_in_ch = f * 4  # 128
        vit_img_size = img_size // 8
        self.vit = ViTEncoder(
            in_channels=vit_in_ch,
            img_size=vit_img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        # 将 ViT 输出映射回 CNN 通道维度 + 恢复 3D 形状
        self.vit_proj = nn.Linear(hidden_size, f * 8)
        self.vit_img_size = vit_img_size

        # ── CNN 解码器 ──
        self.up3 = DecoderBlock(f * 8, f * 4, f * 4)   # 256 → skip=128 → out=128
        self.up2 = DecoderBlock(f * 4, f * 2, f * 2)   # 128 → skip=64  → out=64
        self.up1 = DecoderBlock(f * 2, f, f)           # 64  → skip=32  → out=32

        self.seg_head = nn.Conv3d(f, num_classes, kernel_size=1)
        self._init_weights_cnn()

    def _init_weights_cnn(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # CNN 编码
        e1 = self.enc1(x)           # (B, 32,  D,   H,   W  )
        e2 = self.enc2(self.pool1(e1))  # (B, 64,  D/2, H/2, W/2)
        e3 = self.enc3(self.pool2(e2))  # (B, 128, D/4, H/4, W/4)

        # ViT 瓶颈
        vit_input = self.pool3(e3)      # (B, 128, D/8, H/8, W/8)
        vit_out = self.vit(vit_input)   # (B, N+1, hidden)

        # 去掉 [CLS] token, 只取 patch tokens
        patch_tokens = vit_out[:, 1:, :]        # (B, N, hidden)
        patch_tokens = self.vit_proj(patch_tokens)  # (B, N, f*8)

        # 恢复 3D 空间形状
        d = h = w = self.vit_img_size
        n = d * h * w
        # 如果 patch_tokens 的序列长度多于 d*h*w，截断; 少则报错
        patch_tokens = patch_tokens[:, :n, :]
        feat = patch_tokens.transpose(1, 2).reshape(B, -1, d, h, w)

        # 解码
        d3 = self.up3(feat, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.seg_head(d1)


# ═══════════════════════════════════════════════════════════════
#  工厂函数
# ═══════════════════════════════════════════════════════════════


def build_model(args) -> nn.Module:
    """
    根据 args.model_name 构建对应模型实例。

    Parameters
    ----------
    args : argparse.Namespace (来自 param_set.py)
        需要的字段: model_name, in_channels, num_classes, base_filters,
        attention_gate_channels, vit_img_size, vit_patch_size, vit_hidden_size,
        vit_num_heads, vit_num_layers, vit_mlp_dim, vit_dropout

    Returns
    -------
    nn.Module
    """
    model_map = {
        "nnunet": nnUNet,
        "attention_unet": AttentionUNet,
        "transunet": TransUNet,
    }

    name = args.model_name
    if name not in model_map:
        raise ValueError(f"未知模型: {name}, 可选: {list(model_map.keys())}")

    if name == "nnunet":
        model = nnUNet(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            base_filters=args.base_filters,
        )
    elif name == "attention_unet":
        model = AttentionUNet(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            base_filters=args.base_filters,
            attention_gate_channels=args.attention_gate_channels,
        )
    elif name == "transunet":
        model = TransUNet(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            img_size=args.vit_img_size,
            patch_size=args.vit_patch_size,
            hidden_size=args.vit_hidden_size,
            num_heads=args.vit_num_heads,
            num_layers=args.vit_num_layers,
            mlp_dim=args.vit_mlp_dim,
            dropout=args.vit_dropout,
            base_filters=args.base_filters,
        )

    return model
