"""
model.py -- BrainMRI Segmentation Model Definitions
===================================================
Three 3D medical image segmentation models:
    1. nnU-Net   -- Dynamic U-Net (simplified, no nnU-Net framework dependency)
    2. Attention U-Net -- U-Net with attention gates
    3. TransUNet -- ViT encoder + CNN decoder hybrid architecture
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
#  Shared Components
# ============================================================================


class ConvBlock(nn.Module):
    """Conv3d -> InstanceNorm3d -> LeakyReLU (x2), nnU-Net style."""

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
    """Downsampling: MaxPool3d + ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsampling: transposed conv + concat skip + ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch (odd dimensions cause +1 after upsampling)
        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2,
                       diff_d // 2, diff_d - diff_d // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ============================================================================
#  1. nnU-Net (Simplified Dynamic U-Net)
# ============================================================================


class nnUNet(nn.Module):
    """
    Simplified nnU-Net implementation.

    Key simplifications vs. original nnU-Net:
        - No residual connections / deep supervision
        - Fixed 5-level encoder-decoder structure
        - Retains InstanceNorm + LeakyReLU + Conv3d core style

    Parameters
    ----------
    in_channels : Number of input channels (MRI modalities, default 4)
    num_classes : Number of output classes (including background)
    base_filters : Filter count for the first layer (doubled each level)
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_filters: int = 32,
    ):
        super().__init__()
        f = base_filters  # 32

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)          # -> 32
        self.enc2 = Down(f, f * 2)                     # -> 64
        self.enc3 = Down(f * 2, f * 4)                 # -> 128
        self.enc4 = Down(f * 4, f * 8)                 # -> 256
        self.bottleneck = Down(f * 8, f * 16)          # -> 512

        # Decoder
        self.up4 = Up(f * 16, f * 8, f * 8)            # 512+256 -> 256
        self.up3 = Up(f * 8, f * 4, f * 4)             # 256+128 -> 128
        self.up2 = Up(f * 4, f * 2, f * 2)             # 128+64  -> 64
        self.up1 = Up(f * 2, f, f)                     # 64+32   -> 32

        # Output head
        self.seg_head = nn.Conv3d(f, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)

        # Decode
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.seg_head(d1)


# ============================================================================
#  2. Attention U-Net
# ============================================================================


class AttentionGate(nn.Module):
    """
    3D attention gate.
    Applies soft gating to the skip connection to suppress irrelevant features.
    """

    def __init__(self, f_g: int, f_x: int, f_int: int):
        """
        Parameters
        ----------
        f_g : Gating signal (from lower decoder layer) channel count
        f_x : Skip feature channel count
        f_int : Intermediate compressed channel count
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
        # g1 may have smaller spatial size -> upsample to x size
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="trilinear", align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """
    Attention U-Net (3D).

    Adds an Attention Gate at each decoder level to automatically focus
    on target regions.

    Parameters
    ----------
    in_channels : Input channel count
    num_classes : Output class count
    base_filters : Base filter count
    attention_gate_channels : Attention gate intermediate channel count
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

        # Encoder (same structure as nnUNet)
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = Down(f, f * 2)
        self.enc3 = Down(f * 2, f * 4)
        self.enc4 = Down(f * 4, f * 8)
        self.bottleneck = Down(f * 8, f * 16)

        # Attention gates
        self.ag4 = AttentionGate(f * 16, f * 8, agc)
        self.ag3 = AttentionGate(f * 8, f * 4, agc)
        self.ag2 = AttentionGate(f * 4, f * 2, agc)
        self.ag1 = AttentionGate(f * 2, f, agc)

        # Decoder
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

        # Gated skip connections
        g4 = self.ag4(b, e4)
        d4 = self.up4(b, g4)

        g3 = self.ag3(d4, e3)
        d3 = self.up3(d4, g3)

        g2 = self.ag2(d3, e2)
        d2 = self.up2(d3, g2)

        g1 = self.ag1(d2, e1)
        d1 = self.up1(d2, g1)

        return self.seg_head(d1)


# ============================================================================
#  3. TransUNet
# ============================================================================


class PatchEmbedding3D(nn.Module):
    """Split 3D volume into patches and linearly project to a token sequence."""

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
            f"Input size {D}x{H}x{W} does not match img_size={self.img_size}"
        )
        x = self.proj(x)                            # (B, hidden, D/p, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)            # (B, N, hidden)
        return x


class TransformerEncoderBlock(nn.Module):
    """Standard ViT Transformer Encoder Block."""

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
    """ViT Encoder: PatchEmbed + [CLS] + Positional Encoding + N Transformer layers."""

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
    """TransUNet decoder block: upsample + concat skip + convolution."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Size alignment
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
    TransUNet (3D).

    Encoder uses 3 CNN levels for multi-scale features;
    bottleneck uses ViT for global context;
    decoder upsamples back to original resolution.

    Parameters
    ----------
    in_channels : Input channel count
    num_classes : Output class count
    img_size : ViT input volume size (must be divisible by patch_size)
    patch_size : ViT patch size
    hidden_size : ViT hidden dimension
    num_heads : Number of ViT attention heads
    num_layers : Number of ViT Transformer layers
    mlp_dim : ViT MLP intermediate dimension
    dropout : ViT dropout rate
    base_filters : CNN encoder base channel count
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

        # -- CNN encoder (3 levels) --
        self.enc1 = ConvBlock(in_channels, f)       # 32
        self.pool1 = nn.MaxPool3d(2, 2)
        self.enc2 = ConvBlock(f, f * 2)              # 64
        self.pool2 = nn.MaxPool3d(2, 2)
        self.enc3 = ConvBlock(f * 2, f * 4)          # 128
        self.pool3 = nn.MaxPool3d(2, 2)

        # -- ViT bottleneck --
        # After 3 poolings, size = img_size // 8
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
        # Project ViT output back to CNN channel dimension + restore 3D shape
        self.vit_proj = nn.Linear(hidden_size, f * 8)
        self.vit_img_size = vit_img_size

        # -- CNN decoder --
        self.up3 = DecoderBlock(f * 8, f * 4, f * 4)   # 256 -> skip=128 -> out=128
        self.up2 = DecoderBlock(f * 4, f * 2, f * 2)   # 128 -> skip=64  -> out=64
        self.up1 = DecoderBlock(f * 2, f, f)           # 64  -> skip=32  -> out=32

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

        # CNN encode
        e1 = self.enc1(x)           # (B, 32,  D,   H,   W  )
        e2 = self.enc2(self.pool1(e1))  # (B, 64,  D/2, H/2, W/2)
        e3 = self.enc3(self.pool2(e2))  # (B, 128, D/4, H/4, W/4)

        # ViT bottleneck
        vit_input = self.pool3(e3)      # (B, 128, D/8, H/8, W/8)
        vit_out = self.vit(vit_input)   # (B, N+1, hidden)

        # Drop [CLS] token, keep only patch tokens
        patch_tokens = vit_out[:, 1:, :]        # (B, N, hidden)
        patch_tokens = self.vit_proj(patch_tokens)  # (B, N, f*8)

        # Restore 3D spatial shape
        d = h = w = self.vit_img_size
        n = d * h * w
        # Truncate if token count exceeds d*h*w; raise if fewer
        patch_tokens = patch_tokens[:, :n, :]
        feat = patch_tokens.transpose(1, 2).reshape(B, -1, d, h, w)

        # Decode
        d3 = self.up3(feat, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.seg_head(d1)


# ============================================================================
#  Factory Function
# ============================================================================


def build_model(args) -> nn.Module:
    """
    Build a model instance based on args.model_name.

    Parameters
    ----------
    args : argparse.Namespace (from param_set.py)
        Required fields: model_name, in_channels, num_classes, base_filters,
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
        raise ValueError(f"Unknown model: {name}, available: {list(model_map.keys())}")

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
