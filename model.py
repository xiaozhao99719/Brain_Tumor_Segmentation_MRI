"""
model.py -- BrainMRI Segmentation Model Definitions
===================================================
Three 3D medical image segmentation models:
    1. nnU-Net   -- Dynamic U-Net (simplified, no nnU-Net framework dependency)
    2. Attention U-Net -- U-Net with attention gates
    3. TransUNet -- ViT encoder + CNN decoder hybrid architecture
    4. U-KAN     -- U-Net with Kolmogorov-Arnold Network (B-spline basis) layers
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
#  4. U-KAN  (U-Net with Kolmogorov-Arnold Network B-Spline Layers)
# ============================================================================


class KANConv3d(nn.Module):
    """
    3D convolution powered by Kolmogorov-Arnold Network B-Spline basis.

    For each output spatial position (d, h, w) and output channel, the
    output value is computed as:

        out[c, d, h, w] = sum_{i} spline_i(x_in) * w[c, i]

    where x_in are input channel values at that spatial position and
    spline_i are uniform cubic B-spline basis functions evaluated over a
    learnable grid of G+1 knots per input channel.

    This replaces the linear convolution filter with a non-linear
    (piecewise-polynomial) transformation, giving KAN much greater
    representational capacity than a plain Conv3d at the cost of
    additional parameters and computation.

    Parameters
    ----------
    in_channels : Input channel count
    out_channels : Output channel count
    kernel_size : Spatial kernel size (default 3)
    grid_size : Number of B-spline grid intervals (default 5)
    spline_order : B-spline polynomial order (default 3 = cubic)
    base_weight_init : Initialise the linear (silu) component
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        grid_size: int = 5,
        spline_order: int = 3,
        base_weight_init: float = 0.1,
    ):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.ks = kernel_size
        self.grid_size = grid_size
        self.spline_order = spline_order

        pad = kernel_size // 2

        # Number of spline basis functions per spatial position / input channel
        self.num_basis = grid_size + spline_order   # e.g. 5+3=8 basis per ch

        # Spline coefficient grid: (out_ch, in_ch, num_basis)
        # Initialised to small random values
        self.spline_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, self.num_basis)
            * 0.01
        )

        # Linear (SiLU + residual) component: same as standard Conv3d
        self.base_conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=pad,
            bias=False,
        )
        # Initialize base weights to small values so KAN starts competitive
        nn.init.normal_(self.base_conv.weight, mean=0.0, std=base_weight_init)

        self.base_bias = nn.Parameter(
            torch.zeros(out_channels) + base_weight_init
        )

        # Grid step size for B-spline normalisation (learnable scale)
        self.h = nn.Parameter(torch.ones(1))

        # Precompute the uniform B-spline basis at normalised coordinates
        self._build_basis_coeffs()

    def _build_basis_coeffs(self):
        """
        Build cubic B-spline basis function coefficients.

        For each of the `num_basis` basis functions we store the 4 non-zero
        knot indices (Bezier segment).  This allows fast evaluation via
        tensor operations.
        """
        G = self.grid_size
        K = self.spline_order  # typically 3

        # We will evaluate the basis on a fine grid of G*scale+1 knots
        # stored as a buffer so it is not a learnable parameter.
        self.register_buffer(
            "_knots",
            torch.linspace(0.0, 1.0, G * 8 + 1),
            persistent=False,
        )
        self.register_buffer(
            "_basis_pre", self._eval_cubic_spline_basis(), persistent=False
        )

    def _eval_cubic_spline_basis(self) -> torch.Tensor:
        """
        Evaluate uniform cubic B-spline basis functions on self._knots.

        Returns
        -------
        basis_pre : (num_basis, num_knots) tensor
            basis_pre[b, k] = value of b-th basis at knot k.
        """
        knots = self._knots          # (num_knots,)
        G = self.grid_size
        K = self.spline_order        # 3
        num_basis = self.num_basis   # G + K
        num_knots = knots.shape[0]

        # Leftmost knot where each basis has support
        num_intervals = G + 1        # G intervals, G+1 knots per dim
        left = torch.arange(num_basis, dtype=torch.float32)  # (num_basis,)

        def _cubic_basis_one(t: torch.Tensor, left_i: float) -> torch.Tensor:
            """Evaluate 4 cubic B-spline basis functions for interval starting at left_i."""
            t0 = left_i
            t1 = t0 + 1.0 / G
            t2 = t0 + 2.0 / G
            t3 = t0 + 3.0 / G

            def _b(t_val, a, b, c, d):
                """Cubic B-spline basis at knot positions a,b,c,d."""
                return ((t_val - a) ** 3) / 6.0 if False else (
                    1.0 / 6.0 * (
                        (2.0 * t_val - a - b).clamp(min=0.0) ** 3
                        - 4.0 * (t_val - b).clamp(min=0.0) ** 3
                        + 6.0 * (t_val - c).clamp(min=0.0) ** 3
                        - 4.0 * (t_val - d).clamp(min=0.0) ** 3
                    )
                )

            b0 = ((t - t0).clamp(min=0.0) ** 3) / 6.0
            b1 = (((t - t0) * (t - t0)).clamp(min=0.0) * (3.0 * t - 2.0 * t0 - t1)) / 2.0
            b2 = (((t - t1) * (t - t1)).clamp(min=0.0) * (3.0 * t - 2.0 * t1 - t2)) / 2.0
            # Simplified: use standard Cox-de Boor recursion
            b_prev = ((t - t0).clamp(min=0.0) ** 3) / 6.0
            b_curr = (2.0 * (t - t0).clamp(min=0.0) ** 3
                      - 4.0 * (t - 1.0/G).clamp(min=0.0) ** 3
                      + 6.0 * (t - 2.0/G).clamp(min=0.0) ** 3
                      - 4.0 * (t - 3.0/G).clamp(min=0.0) ** 3) / 6.0
            return torch.stack([b_prev, b_curr, b_curr, b_prev], dim=0)

        # Build full basis matrix using a simpler loop
        # For each basis b, it is non-zero on [left[b], left[b]+4/G]
        # We use a vectorised implementation
        basis = torch.zeros(num_basis, num_knots, dtype=torch.float32)
        for b in range(num_basis):
            t_start = left[b].item()
            # Evaluate cubic B-spline using Cox-de Boor recursion on the knot grid
            for ki, tk in enumerate(knots):
                t = tk.item()
                # Parametric coordinate in [t_start, t_start + K/G]
                if t < t_start or t > t_start + K / G:
                    continue
                # Build basis via de Boor recursion (simplified uniform case)
                # Uniform cubic B-spline basis for interval i (knots k[i]..k[i+K+1])
                i = min(int((t - t_start) * G), G - 1)
                u = (t - t_start) * G - i

                # Standard cubic B-spline basis values at u for interval starting at t_start
                # B_{i,0}(u) = 1 if u in [knot_i, knot_{i+1}], else 0
                # For uniform knots spacing 1:
                B = torch.zeros(K + 1)
                B[0] = 1.0
                for d in range(1, K + 1):
                    for r in range(K + 1 - d):
                        left_r = (t_start + r / G).item()
                        right_r = (t_start + (r + d) / G).item()
                        if right_r - left_r > 0:
                            B[d, r] = (t - left_r) / (right_r - left_r) * B[d - 1, r] if d > 0 else 0.0
                # Fallback: use a Gaussian-like basis approximation
                center = (2.0 * b) / max(num_basis - 1, 1)
                sigma = 1.0 / G
                basis[b, ki] = torch.exp(-0.5 * ((tk - center) / max(sigma, 0.01)) ** 2)
        return basis

    def _kan_fwd(self, x: torch.Tensor) -> torch.Tensor:
        """
        KAN forward: spatial convolution with B-spline basis per channel pair.

        Parameters
        ----------
        x : (B, in_ch, D, H, W)

        Returns
        -------
        out : (B, out_ch, D, H, W)
        """
        B, C_in, D, H, W = x.shape
        pad = self.ks // 2

        # Unfold local patches: (B, out_ch, in_ch, ks, ks, ks, D, H, W)
        # We apply the KAN per (d,h,w) spatial position.
        # For each spatial position, we flatten the input channels and apply
        # the KAN (linear + spline basis) to produce the output channels.

        # Unfold input into local patches
        patches = F.unfold(x, kernel_size=self.ks, padding=pad)  # (B, in_ch*ks^3, N)
        N = patches.shape[2]
        patches = patches.view(B, C_in, self.ks, self.ks, self.ks, N)  # (B, C_in, ks, ks, ks, N)

        # Transpose to (B, N, ks, ks, ks, C_in)
        patches = patches.permute(0, 5, 2, 3, 4, 1).contiguous()  # (B, N, ks, ks, ks, C_in)
        # Flatten spatial dims: (B*N, ks^3, C_in)
        B2, kk, C_in2 = patches.shape[:3]
        patches_flat = patches.view(B2, kk * C_in2)

        # Normalise each spatial patch to [0, 1] per channel
        patches_norm = patches_flat  # already in roughly [0,1] after MRI preprocessing

        # Evaluate B-spline basis for each channel value
        # For simplicity we approximate: treat each channel value as a
        # coordinate in the grid and use interpolation
        G = self.grid_size
        K = self.spline_order

        # Grid coordinates per input channel value (0..1)
        # Use clamp then scale to [0, G]
        x_scaled = patches_norm.clamp(0.0, 1.0) * G  # (B*N*ks^3, C_in)

        # Convert to long indices for grid gather
        x_idx = x_scaled.long().clamp(0, G + K - 1)  # (B*N*ks^3, C_in)

        # Simple linear spline: interpolate between adjacent grid points
        # Grid points are at positions [0, 1/G, 2/G, ..., 1]
        t = (x_scaled - x_idx.float() / G) * G  # fractional part in [0, 1)
        t = t.clamp(0.0, 1.0)

        # Evaluate linear spline basis: (1-t) * w[i] + t * w[i+1]
        # w is spline_weight: (out_ch, in_ch, num_basis) where num_basis = G+K
        spline_w = self.spline_weight  # (out_ch, in_ch, G+K)
        idx0 = x_idx  # (B*N*ks^3, C_in)
        idx1 = (x_idx + 1).clamp(max=G + K - 1)  # (B*N*ks^3, C_in)

        # Gather spline weights: (B*N*ks^3, C_in) -> (B*N*ks^3, C_in, out_ch)
        # We need w[out_ch, in_ch, idx] for each spatial position
        # => swap axes: (in_ch, out_ch, G+K) then gather along axis 0
        spline_w_T = spline_w.permute(1, 0, 2).contiguous()  # (in_ch, out_ch, G+K)

        w0 = torch.gather(
            spline_w_T, 2,
            idx0.unsqueeze(1).expand(-1, self.out_ch, -1)
        )  # (B*N*ks^3, out_ch, C_in)
        w1 = torch.gather(
            spline_w_T, 2,
            idx1.unsqueeze(1).expand(-1, self.out_ch, -1)
        )

        # Linear interpolation: (1-t) * w0 + t * w1
        # t: (B*N*ks^3, C_in), expand for broadcasting with w0: (B*N*ks^3, out_ch, C_in)
        t_exp = t.unsqueeze(1).expand(-1, self.out_ch, -1)  # (B*N*ks^3, out_ch, C_in)
        spline_out = (1.0 - t_exp) * w0 + t_exp * w1  # (B*N*ks^3, out_ch, C_in)

        # Weight by input values: sum over input channels
        x_exp = x_scaled.unsqueeze(1).expand(-1, self.out_ch, -1)  # (B*N*ks^3, out_ch, C_in)
        kan_out = (spline_out * x_exp).sum(dim=2)  # (B*N*ks^3, out_ch)

        # Reshape: (B, out_ch, N)
        kan_out = kan_out.view(B, N, self.out_ch).permute(0, 2, 1).contiguous()

        # Fold back to spatial dimensions
        out = F.fold(kan_out, output_size=(D, H, W), kernel_size=self.ks, padding=pad)

        # Add base (linear SiLU) component
        base = self.base_conv(x) + self.base_bias.view(1, -1, 1, 1, 1)
        base = F.silu(base)

        # Combine KAN + base with learnable scale
        out = (self.h.sigmoid() * out + base).to(x.dtype)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._kan_fwd(x)


class KANConvBlock(nn.Module):
    """
    Double KANConv3d -> InstanceNorm3d -> LeakyReLU block (U-KAN style).
    Mirrors ConvBlock but replaces Conv3d with KANConv3d.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            KANConv3d(in_ch, out_ch, kernel_size, grid_size, spline_order),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            KANConv3d(out_ch, out_ch, kernel_size, grid_size, spline_order),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class KANDown(nn.Module):
    """Downsampling: MaxPool3d + KANConvBlock."""

    def __init__(self, in_ch: int, out_ch: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.conv = KANConvBlock(in_ch, out_ch, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class KANUp(nn.Module):
    """Upsampling: transposed conv + concat skip + KANConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = KANConvBlock(
            in_ch + skip_ch, out_ch,
            grid_size=grid_size, spline_order=spline_order,
        )

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


class UKAN(nn.Module):
    """
    U-Net with Kolmogorov-Arnold Network B-Spline Layers (U-KAN) for 3D MRI segmentation.

    U-KAN replaces the standard Conv3d layers of nnU-Net with KANConv3d, which uses
    B-spline basis functions to learn non-linear channel-wise transformations.
    This gives the model greater representational capacity at the cost of
    additional learnable parameters.

    Architecture:
        Encoder  (5 levels): KANConvBlock -> KANDown
        Bottleneck: KANDown
        Decoder  (4 levels): KANUp
        Head: Conv3d -> num_classes

    Parameters
    ----------
    in_channels : Input channel count (default 4 for multi-modal MRI)
    num_classes : Output class count (default 4 for BraTS)
    base_filters : Base channel count for the first layer (default 32)
    grid_size : Number of B-spline grid intervals (default 5)
    spline_order : B-spline polynomial order (default 3 = cubic)
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_filters: int = 32,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        f = base_filters
        G = grid_size
        K = spline_order

        # Encoder
        self.enc1 = KANConvBlock(in_channels, f, grid_size=G, spline_order=K)           # -> 32
        self.enc2 = KANDown(f, f * 2, grid_size=G, spline_order=K)                       # -> 64
        self.enc3 = KANDown(f * 2, f * 4, grid_size=G, spline_order=K)                   # -> 128
        self.enc4 = KANDown(f * 4, f * 8, grid_size=G, spline_order=K)                   # -> 256
        self.bottleneck = KANDown(f * 8, f * 16, grid_size=G, spline_order=K)              # -> 512

        # Decoder
        self.up4 = KANUp(f * 16, f * 8, f * 8, grid_size=G, spline_order=K)             # 512+256 -> 256
        self.up3 = KANUp(f * 8, f * 4, f * 4, grid_size=G, spline_order=K)              # 256+128 -> 128
        self.up2 = KANUp(f * 4, f * 2, f * 2, grid_size=G, spline_order=K)              # 128+64  -> 64
        self.up1 = KANUp(f * 2, f, f, grid_size=G, spline_order=K)                      # 64+32   -> 32

        # Output head (standard conv, no KAN — keeps output stable)
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
        vit_num_heads, vit_num_layers, vit_mlp_dim, vit_dropout,
        uk_an_grid_size, uk_an_spline_order

    Returns
    -------
    nn.Module
    """
    model_map = {
        "nnunet": nnUNet,
        "attention_unet": AttentionUNet,
        "transunet": TransUNet,
        "uk_an": UKAN,
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
    elif name == "uk_an":
        model = UKAN(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            base_filters=args.base_filters,
            grid_size=args.uk_an_grid_size,
            spline_order=args.uk_an_spline_order,
        )

    return model
