"""
param_set.py — BrainMRI 全局参数配置
======================================
集中管理 data_load / model / train / test 所有可配置参数，
通过 argparse 统一解析，其他模块从本文件导入使用。
"""

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    """构建全局参数解析器，返回 ArgumentParser 对象。"""
    parser = argparse.ArgumentParser(description="BrainMRI Segmentation — 全局参数配置")

    # ── 数据相关 ──────────────────────────────────────────────
    group_data = parser.add_argument_group("Data")
    group_data.add_argument("--data_root", type=str,
                            default="E:/python/BrainMRI",
                            help="原始数据根目录")
    group_data.add_argument("--cache_root", type=str,
                            default=None,
                            help="预处理缓存目录 (默认: <data_root>/preprocessed)")
    group_data.add_argument("--target_spacing", type=float,
                            default=1.0,
                            help="各向同性重采样目标间距 (mm)")
    group_data.add_argument("--n4_enabled", type=int,
                            default=1, choices=[0, 1],
                            help="是否启用 N4 偏置场校正 (1=是, 0=否)")
    group_data.add_argument("--use_cache", type=int,
                            default=1, choices=[0, 1],
                            help="是否使用磁盘缓存加载预处理数据 (1=是, 0=否)")
    group_data.add_argument("--preprocess_workers", type=int,
                            default=4,
                            help="离线预处理并行进程数")
    group_data.add_argument("--preprocess_force", action="store_true",
                            help="强制重新预处理，忽略已有缓存")
    group_data.add_argument("--num_classes", type=int,
                            default=4,
                            help="分割类别数 (含背景: 0=BG, 1=NCR, 2=ED, 3=ET)")

    # ── 模型相关 ──────────────────────────────────────────────
    group_model = parser.add_argument_group("Model")
    group_model.add_argument("--model_name", type=str,
                             default="nnunet",
                             choices=["nnunet", "attention_unet", "transunet"],
                             help="选择模型架构")
    group_model.add_argument("--in_channels", type=int,
                             default=4,
                             help="输入通道数 (MRI 模态数)")
    group_model.add_argument("--base_filters", type=int,
                             default=32,
                             help="U-Net / nnU-Net 基础滤波器数")
    group_model.add_argument("--attention_gate_channels", type=int,
                             default=128,
                             help="Attention U-Net 门控中间通道数")
    # TransUNet 专用
    group_model.add_argument("--vit_img_size", type=int,
                             default=128,
                             help="TransUNet ViT 输入图像尺寸 (需与 patch 大小匹配)")
    group_model.add_argument("--vit_patch_size", type=int,
                             default=16,
                             help="TransUNet ViT patch 大小")
    group_model.add_argument("--vit_hidden_size", type=int,
                             default=768,
                             help="TransUNet ViT 隐层维度")
    group_model.add_argument("--vit_num_heads", type=int,
                             default=12,
                             help="TransUNet ViT 注意力头数")
    group_model.add_argument("--vit_num_layers", type=int,
                             default=12,
                             help="TransUNet ViT Transformer 层数")
    group_model.add_argument("--vit_mlp_dim", type=int,
                             default=3072,
                             help="TransUNet ViT MLP 中间维度")
    group_model.add_argument("--vit_dropout", type=float,
                             default=0.1,
                             help="TransUNet ViT dropout 率")

    # ── 训练相关 ──────────────────────────────────────────────
    group_train = parser.add_argument_group("Training")
    group_train.add_argument("--epochs", type=int,
                             default=150,
                             help="训练轮数")
    group_train.add_argument("--batch_size", type=int,
                             default=1,
                             help="训练批大小")
    group_train.add_argument("--lr", type=float,
                             default=1e-3,
                             help="初始学习率")
    group_train.add_argument("--weight_decay", type=float,
                             default=1e-4,
                             help="AdamW 权重衰减")
    group_train.add_argument("--lr_scheduler", type=str,
                             default="cosine",
                             choices=["cosine", "step", "none"],
                             help="学习率调度器")
    group_train.add_argument("--step_lr_step_size", type=int,
                             default=30,
                             help="StepLR 步长 (仅 step 调度器)")
    group_train.add_argument("--step_lr_gamma", type=float,
                             default=0.1,
                             help="StepLR 衰减系数 (仅 step 调度器)")
    group_train.add_argument("--loss_fn", type=str,
                             default="dice_ce",
                             choices=["dice_ce", "dice", "ce", "focal"],
                             help="损失函数组合")
    group_train.add_argument("--focal_gamma", type=float,
                             default=2.0,
                             help="Focal Loss gamma 参数")
    group_train.add_argument("--dice_smooth", type=float,
                             default=1.0,
                             help="Dice Loss 平滑项")
    group_train.add_argument("--num_workers", type=int,
                             default=4,
                             help="DataLoader 工作进程数")
    group_train.add_argument("--pin_memory", type=int,
                             default=0, choices=[0, 1],
                             help="是否 pin_memory")
    group_train.add_argument("--prefetch_factor", type=int,
                             default=2,
                             help="DataLoader prefetch_factor")

    # ── 3D patch 训练 ────────────────────────────────────────
    group_patch = parser.add_argument_group("3D Patch")
    group_patch.add_argument("--patch_size", type=int, nargs=3,
                             default=[96, 96, 96],
                             help="3D 训练 patch 大小 (D H W)")
    group_patch.add_argument("--patch_overlap", type=float,
                             default=0.25,
                             help="滑窗推理重叠比例 (0~0.5)")
    group_patch.add_argument("--random_patch", type=int,
                             default=1, choices=[0, 1],
                             help="训练时随机裁剪 patch (1=是)")

    # ── 测试 / 评估相关 ──────────────────────────────────────
    group_test = parser.add_argument_group("Test / Evaluation")
    group_test.add_argument("--test_split", type=str,
                            default="test",
                            help="评估所用数据集划分 (train/val/test)")
    group_test.add_argument("--save_pred", action="store_true",
                            help="是否保存预测结果为 NIfTI")
    group_test.add_argument("--tta", action="store_true",
                            help="测试时增强 (4 翻转)")

    # ── 输出 / 日志 ──────────────────────────────────────────
    group_out = parser.add_argument_group("Output / Logging")
    group_out.add_argument("--output_dir", type=str,
                           default="E:/python/BrainMRI/output",
                           help="输出目录 (权重/日志/预测)")
    group_out.add_argument("--save_every", type=int,
                           default=10,
                           help="每 N 个 epoch 保存一次检查点")
    group_out.add_argument("--log_every", type=int,
                           default=10,
                           help="每 N 个 batch 打印训练日志")
    group_out.add_argument("--resume_ckpt", type=str,
                           default=None,
                           help="恢复训练的检查点路径")
    group_out.add_argument("--seed", type=int,
                           default=42,
                           help="随机种子")

    # ── 硬件 ─────────────────────────────────────────────────
    group_hw = parser.add_argument_group("Hardware")
    group_hw.add_argument("--gpu", type=int,
                          default=0,
                          help="使用的 GPU 编号 (-1 表示 CPU)")
    group_hw.add_argument("--amp", type=int,
                          default=1, choices=[0, 1],
                          help="是否使用混合精度训练 (AMP)")

    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """解析参数并做一些后处理。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    # 自动推导 cache_root
    if args.cache_root is None:
        args.cache_root = os.path.join(args.data_root, "preprocessed")

    # AMP 与 GPU 兼容
    if args.gpu < 0:
        args.device = "cpu"
        args.amp = 0
    else:
        import torch
        args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        if args.device == "cpu":
            args.amp = 0

    return args
