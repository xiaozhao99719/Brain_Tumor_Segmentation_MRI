"""
main.py — BrainMRI 分割项目主入口
====================================
负责调用各模块的函数, 按流程执行:
  1. 解析参数 (param_set)
  2. 离线预处理 (data_load)  — 可选, 仅在缓存不存在时需要
  3. 训练 (train)
  4. 测试评估 (test)

用法示例:
  # 完整流程: 预处理 + 训练 + 测试
  python main.py --mode all --model_name nnunet

  # 仅训练
  python main.py --mode train --model_name attention_unet

  # 仅测试
  python main.py --mode test --model_name transunet

  # 仅预处理
  python main.py --mode preprocess --preprocess_workers 8

  # 训练 + 测试 (跳过预处理)
  python main.py --mode train_test --model_name nnunet --epochs 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from param_set import parse_args


# ═══════════════════════════════════════════════════════════════
#  子流程函数
# ═══════════════════════════════════════════════════════════════


def run_preprocess(args) -> None:
    """调用 data_load 执行离线预处理。"""
    from data_load import preprocess_split

    print("\n" + "=" * 70)
    print("  步骤: 数据预处理")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        preprocess_split(
            split=split,
            target_spacing=args.target_spacing,
            workers=args.preprocess_workers,
            force=args.preprocess_force,
        )

    print("  预处理完成!\n")


def run_train(args) -> None:
    """调用 train 模块执行训练。"""
    from train import train as train_fn

    print("\n" + "=" * 70)
    print("  步骤: 模型训练")
    print("=" * 70)

    t0 = time.time()
    model = train_fn(args)
    elapsed = time.time() - t0
    print(f"  训练耗时: {elapsed / 60:.1f} 分钟\n")


def run_test(args) -> None:
    """调用 test 模块执行评估。"""
    from test import test as test_fn

    print("\n" + "=" * 70)
    print("  步骤: 模型测试")
    print("=" * 70)

    t0 = time.time()
    metrics = test_fn(args)
    elapsed = time.time() - t0
    print(f"  测试耗时: {elapsed / 60:.1f} 分钟\n")


def run_info(args) -> None:
    """调用 data_load 打印数据集信息。"""
    from data_load import print_dataset_info

    print_dataset_info(split="all", target_spacing=args.target_spacing)


# ═══════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════


def main() -> None:
    """主入口: 解析参数 → 按模式执行对应流程。"""

    # 解析全局参数
    args = parse_args()

    # 添加运行模式参数 (不放入 param_set 避免与其他模块耦合)
    # 使用 sys.argv 中的 --mode 或默认 "all"
    mode = "all"
    for i, arg in enumerate(sys.argv):
        if arg == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
            break

    valid_modes = ["all", "preprocess", "train", "test", "train_test", "info"]
    if mode not in valid_modes:
        print(f"错误: 未知模式 '{mode}', 可选: {valid_modes}")
        sys.exit(1)

    # ── 环境信息 ──
    print("=" * 70)
    print("  BrainMRI 分割项目")
    print("=" * 70)
    print(f"  运行模式 : {mode}")
    print(f"  模型     : {args.model_name}")
    print(f"  设备     : {args.device}")
    print(f"  AMP      : {'启用' if args.amp == 1 else '关闭'}")
    print(f"  数据根目录: {args.data_root}")
    print(f"  缓存目录  : {args.cache_root}")
    print(f"  输出目录  : {args.output_dir}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  学习率   : {args.lr}")
    print(f"  Patch Size: {args.patch_size}")
    if torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(args.gpu)}")
    print("=" * 70)

    # ── 执行流程 ──
    if mode == "info":
        run_info(args)
        return

    if mode in ("all", "preprocess"):
        run_preprocess(args)

    if mode in ("all", "train", "train_test"):
        run_train(args)

    if mode in ("all", "test", "train_test"):
        run_test(args)

    print("\n" + "=" * 70)
    print("  全部流程完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
