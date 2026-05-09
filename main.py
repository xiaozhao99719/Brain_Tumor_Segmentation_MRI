"""
main.py -- BrainMRI Segmentation Project Main Entry
====================================================
Orchestrates the full pipeline by calling module functions in order:
    1. Parse arguments (param_set)
    2. Offline preprocessing (data_load)  -- optional, only needed if cache is missing
    3. Training (train)
    4. Testing / evaluation (test)

Usage examples:
    # Full pipeline: preprocess + train + test
    python main.py --mode all --model_name nnunet

    # Training only
    python main.py --mode train --model_name attention_unet

    # Testing only
    python main.py --mode test --model_name transunet

    # Preprocessing only
    python main.py --mode preprocess --preprocess_workers 8

    # Train + test (skip preprocessing)
    python main.py --mode train_test --model_name nnunet --epochs 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from param_set import parse_args


# ============================================================================
#  Sub-pipeline Functions
# ============================================================================


def run_preprocess(args) -> None:
    """Run offline preprocessing via data_load."""
    from data_load import preprocess_split

    print("\n" + "=" * 70)
    print("  Step: Data Preprocessing")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        preprocess_split(
            split=split,
            target_spacing=args.target_spacing,
            workers=args.preprocess_workers,
            force=args.preprocess_force,
        )

    print("  Preprocessing done!\n")


def run_train(args) -> None:
    """Run training via the train module."""
    from train import train as train_fn

    print("\n" + "=" * 70)
    print("  Step: Model Training")
    print("=" * 70)

    t0 = time.time()
    model = train_fn(args)
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed / 60:.1f} minutes\n")


def run_test(args) -> None:
    """Run evaluation via the test module."""
    from test import test as test_fn

    print("\n" + "=" * 70)
    print("  Step: Model Testing")
    print("=" * 70)

    t0 = time.time()
    metrics = test_fn(args)
    elapsed = time.time() - t0
    print(f"  Testing time: {elapsed / 60:.1f} minutes\n")


def run_info(args) -> None:
    """Print dataset info via data_load."""
    from data_load import print_dataset_info

    print_dataset_info(split="all", target_spacing=args.target_spacing)


# ============================================================================
#  Main Entry Point
# ============================================================================


def main() -> None:
    """Main entry: parse args -> execute pipeline by mode."""

    # Parse global arguments
    args = parse_args()

    # Extract --mode from sys.argv (kept here to avoid coupling param_set with main)
    mode = "all"
    for i, arg in enumerate(sys.argv):
        if arg == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
            break

    valid_modes = ["all", "preprocess", "train", "test", "train_test", "info"]
    if mode not in valid_modes:
        print(f"Error: unknown mode '{mode}', valid modes: {valid_modes}")
        sys.exit(1)

    # -- Environment info --
    print("=" * 70)
    print("  BrainMRI Segmentation Project")
    print("=" * 70)
    print(f"  Mode       : {mode}")
    print(f"  Model      : {args.model_name}")
    print(f"  Device     : {args.device}")
    print(f"  AMP        : {'ON' if args.amp == 1 else 'OFF'}")
    print(f"  Data root  : {args.data_root}")
    print(f"  Cache root : {args.cache_root}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.effective_lr}")
    print(f"  Patch size : {args.patch_size}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(args.gpu)}")
    print("=" * 70)

    # -- Execute pipeline --
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
    print("  All steps complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
