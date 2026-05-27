"""
独立可视化脚本
不依赖param_set.py/main.py，内部自包含所有参数
从测试集随机选取5个样本，可视化对比GT和预测结果
"""

import os
import sys
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import random

# 添加项目路径，导入model.py中的模型定义
PROJECT_DIR = r'D:\PythonProgram\BrainSeg\Class1'
sys.path.insert(0, PROJECT_DIR)
from model import get_model


###########################################
# 内部配置（修改这里以适配您的数据）
###########################################

# 路径配置
BEST_MODEL_PATH = r'D:\PythonProgram\BrainSeg\Class1\checkpoints\best_model.pth'  # 最佳模型路径
TEST_IMAGE_DIR = r'D:\PythonProgram\BrainSeg\test\image'  # 测试集图像路径
TEST_LABEL_DIR = r'D:\PythonProgram\BrainSeg\test\labels'  # 测试集标签路径
OUTPUT_DIR = r'D:\PythonProgram\BrainSeg\Class1\Visable_contrast'  # 输出根路径

# 模型配置（需要与训练时一致）
MODEL_NAME = 'transbts'  # 或 'kiunet'
IN_CHANNELS = 4  # 输入通道数（模态数）
NUM_CLASSES = 2  # 输出类别数
BASE_FILTERS = 32
TARGET_SIZE = (128, 128, 128)  # 与训练时一致
IMG_SIZE = 128  # TransBTS需要的参数

# 可视化配置
NUM_SAMPLES = 5  # 随机选取的样本数
MODALITY_TO_SHOW = 0  # 显示哪个模态（0=t1, 1=t1c, 2=t2, 3=flair，通常flair=3病变最明显）
SLICE_SELECTION = 'middle'  # 'middle'=中间切片, 'max_foreground'=前景最多的切片

# 颜色配置
GT_COLOR = [0, 1, 0]  # GT颜色 (绿色)
PRED_COLOR = [1, 0, 0]  # 预测颜色 (红色)
ALPHA = 0.4  # 透明度

# 设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


###########################################
# 预处理函数（简化版）
###########################################

def preprocess_image(image_path, target_size):
    """预处理图像"""
    img = nib.load(image_path)
    data = img.get_fdata()
    
    # 调整尺寸（简化版）
    from scipy import ndimage
    zoom_factors = [target_size[i] / data.shape[i] for i in range(3)]
    if len(data.shape) == 4:
        resized = np.zeros((target_size[0], target_size[1], target_size[2], data.shape[3]))
        for m in range(data.shape[3]):
            resized[:, :, :, m] = ndimage.zoom(data[:, :, :, m], zoom_factors, order=3)
    else:
        resized = ndimage.zoom(data, zoom_factors + [1] if len(data.shape)==4 else zoom_factors, order=3)
    
    # Z-score标准化
    if len(resized.shape) == 4:
        for m in range(resized.shape[3]):
            mean = np.mean(resized[:, :, :, m])
            std = np.std(resized[:, :, :, m])
            resized[:, :, :, m] = (resized[:, :, :, m] - mean) / (std + 1e-8)
    else:
        mean = np.mean(resized)
        std = np.std(resized)
        resized = (resized - mean) / (std + 1e-8)
    
    # 转换为PyTorch格式 (C, D, H, W)
    if len(resized.shape) == 4:
        resized = np.transpose(resized, (3, 2, 0, 1))
    else:
        resized = np.expand_dims(resized, axis=0)
        resized = np.transpose(resized, (0, 3, 1, 2))
    
    return torch.from_numpy(resized).float()


def preprocess_label(label_path, target_size):
    """预处理标签"""
    img = nib.load(label_path)
    data = img.get_fdata()
    
    # 调整尺寸
    from scipy import ndimage
    zoom_factors = [target_size[i] / data.shape[i] for i in range(3)]
    resized = ndimage.zoom(data, zoom_factors, order=0)  # 最近邻
    
    # 二值化
    resized = (resized > 0).astype(np.int64)
    
    # 转换为PyTorch格式 (D, H, W)
    resized = np.transpose(resized, (2, 0, 1))
    
    return torch.from_numpy(resized).long()


###########################################
# 可视化函数
###########################################

def select_slice_2d(volume_3d, mode='middle'):
    """
    从3D体中选择一个2D切片用于可视化
    
    Args:
        volume_3d: 3D numpy数组 (D, H, W)
        mode: 'middle'=中间切片, 'max_foreground'=前景最多的切片
    
    Returns:
        slice_index: 切片索引
    """
    if mode == 'middle':
        return volume_3d.shape[0] // 2
    elif mode == 'max_foreground':
        # 找到前景最多的切片
        foreground_counts = [np.sum(volume_3d[i] > 0) for i in range(volume_3d.shape[0])]
        return np.argmax(foreground_counts)
    else:
        return volume_3d.shape[0] // 2


def visualize_slice(image_slice, gt_slice, pred_slice, filename, output_path):
    """
    可视化单个切片
    
    Args:
        image_slice: 2D图像切片 (H, W)，已归一化到[0,1]
        gt_slice: 2D GT切片 (H, W)
        pred_slice: 2D预测切片 (H, W)
        filename: 文件名（用于标题）
        output_path: 输出路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 原图
    axes[0].imshow(image_slice, cmap='gray')
    axes[0].set_title('MRI Image')
    axes[0].axis('off')
    
    # 2. GT叠加
    axes[1].imshow(image_slice, cmap='gray')
    if np.sum(gt_slice) > 0:
        gt_mask = np.ma.masked_where(gt_slice == 0, gt_slice)
        axes[1].imshow(gt_mask, cmap='Greens', alpha=ALPHA, vmin=0, vmax=1)
    axes[1].set_title('GT Overlay')
    axes[1].axis('off')
    
    # 3. 预测叠加
    axes[2].imshow(image_slice, cmap='gray')
    if np.sum(pred_slice) > 0:
        pred_mask = np.ma.masked_where(pred_slice == 0, pred_slice)
        axes[2].imshow(pred_mask, cmap='Reds', alpha=ALPHA, vmin=0, vmax=1)
    axes[2].set_title('Prediction Overlay')
    axes[2].axis('off')
    
    # 添加颜色条说明
    import matplotlib.patches as mpatches
    gt_patch = mpatches.Patch(color=GT_COLOR, label='GT')
    pred_patch = mpatches.Patch(color=PRED_COLOR, label='Prediction')
    axes[2].legend(handles=[gt_patch, pred_patch], loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.suptitle(filename, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


###########################################
# 主函数
###########################################

def main():
    """主函数"""
    print("="*60)
    print(f"独立可视化脚本 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 检查路径
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"✗ 错误: 最佳模型不存在: {BEST_MODEL_PATH}")
        return
    
    if not os.path.exists(TEST_IMAGE_DIR):
        print(f"✗ 错误: 测试集图像目录不存在: {TEST_IMAGE_DIR}")
        return
    
    if not os.path.exists(TEST_LABEL_DIR):
        print(f"✗ 错误: 测试集标签目录不存在: {TEST_LABEL_DIR}")
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ 输出目录: {OUTPUT_DIR}")
    
    # 加载模型（从model.py导入）
    print(f"\n加载模型: {MODEL_NAME}")
    model = get_model(
        MODEL_NAME,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        base_filters=BASE_FILTERS,
        img_size=IMG_SIZE
    )
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    print(f"✓ 模型加载成功 (训练Epoch: {checkpoint['epoch']+1})")
    
    # 获取测试文件列表
    test_files = sorted([f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith('.nii')])
    
    if len(test_files) == 0:
        print(f"✗ 错误: 测试集图像目录中没有.nii文件: {TEST_IMAGE_DIR}")
        return
    
    # 随机选取样本
    if NUM_SAMPLES > len(test_files):
        print(f"⚠ 警告: 测试集只有 {len(test_files)} 个样本，将全部使用")
        selected_files = test_files
    else:
        selected_files = random.sample(test_files, NUM_SAMPLES)
    
    print(f"\n随机选取 {len(selected_files)} 个样本进行可视化")
    print(f"使用的模态: {MODALITY_TO_SHOW}")
    print(f"切片选择: {SLICE_SELECTION}")
    
    # 处理每个样本
    with torch.no_grad():
        for idx, filename in enumerate(selected_files):
            print(f"\n[{idx+1}/{len(selected_files)}] 处理: {filename}")
            
            # 构建文件路径
            image_path = os.path.join(TEST_IMAGE_DIR, filename)
            label_path = os.path.join(TEST_LABEL_DIR, filename)
            
            if not os.path.exists(label_path):
                print(f"  ⚠ 警告: 标签文件不存在，跳过: {label_path}")
                continue
            
            # 预处理
            image_tensor = preprocess_image(image_path, TARGET_SIZE)
            label_tensor = preprocess_label(label_path, TARGET_SIZE)
            
            # 推理
            image_tensor = image_tensor.unsqueeze(0).to(DEVICE)  # (1, C, D, H, W)
            output = model(image_tensor)
            pred = torch.argmax(output, dim=1)  # (1, D, H, W)
            
            # 转换为numpy
            image_np = image_tensor.cpu().numpy()[0]  # (C, D, H, W)
            label_np = label_tensor.cpu().numpy()      # (D, H, W)
            pred_np = pred.cpu().numpy()[0]           # (D, H, W)
            
            # 选择要可视化的模态和切片
            # 图像: (C, D, H, W) -> 选择模态 -> (D, H, W) -> 选择切片 -> (H, W)
            image_modal = image_np[MODALITY_TO_SHOW]  # (D, H, W)
            
            # 选择切片
            slice_idx = select_slice_2d(label_np if np.sum(label_np) > 0 else pred_np, SLICE_SELECTION)
            
            image_slice = image_modal[slice_idx]  # (H, W)
            gt_slice = label_np[slice_idx]        # (H, W)
            pred_slice = pred_np[slice_idx]       # (H, W)
            
            # 归一化图像切片到[0,1]用于显示
            image_slice_norm = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
            
            # 可视化
            output_path = os.path.join(OUTPUT_DIR, f"visual_{filename.replace('.nii', '.png')}")
            visualize_slice(image_slice_norm, gt_slice, pred_slice, filename, output_path)
            print(f"  ✓ 保存可视化结果: {output_path}")
    
    print("\n" + "="*60)
    print(f"可视化完成！结果保存在: {OUTPUT_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
