"""
数据预处理与加载模块
包含：数据校验、预处理、增强、PyTorch数据加载
"""

import os
import numpy as np
import nibabel as nib
from scipy import ndimage
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random


class MedicalImageProcessor:
    """医学影像预处理类"""
    
    def __init__(self, target_spacing=(1.0, 1.0, 1.0), target_size=(128, 128, 128)):
        self.target_spacing = target_spacing
        self.target_size = target_size
    
    def load_nifti(self, file_path):
        """加载NIfTI文件，返回数据和元数据"""
        img = nib.load(file_path)
        data = img.get_fdata()
        spacing = img.header.get_zooms()
        affine = img.affine
        return data, spacing, affine
    
    def resample_to_target_spacing(self, data, current_spacing, target_spacing, order=3):
        """重采样到目标体素间距"""
        # 计算缩放因子
        zoom_factors = [current_spacing[i] / target_spacing[i] for i in range(3)]
        
        # 对多模态数据，只对空间维度重采样
        if len(data.shape) == 4:  # (H, W, D, modalities)
            resampled = np.zeros((int(data.shape[0] * zoom_factors[0]),
                                  int(data.shape[1] * zoom_factors[1]),
                                  int(data.shape[2] * zoom_factors[2]),
                                  data.shape[3]))
            for m in range(data.shape[3]):
                resampled[:, :, :, m] = ndimage.zoom(data[:, :, :, m], zoom_factors, order=order)
        else:  # 3D数据
            resampled = ndimage.zoom(data, zoom_factors, order=order)
        
        return resampled
    
    def resize_to_target(self, data, target_size):
        """调整尺寸到目标大小"""
        if len(data.shape) == 4:  # 多模态
            current_size = data.shape[:3]
            zoom_factors = [target_size[i] / current_size[i] for i in range(3)]
            resized = np.zeros((target_size[0], target_size[1], target_size[2], data.shape[3]))
            for m in range(data.shape[3]):
                resized[:, :, :, m] = ndimage.zoom(data[:, :, :, m], zoom_factors, order=3)
        else:  # 3D
            current_size = data.shape
            zoom_factors = [target_size[i] / current_size[i] for i in range(3)]
            resized = ndimage.zoom(data, zoom_factors, order=3)
        
        return resized
    
    def normalize_intensity(self, data, method='z-score'):
        """图像强度标准化"""
        if len(data.shape) == 4:  # 多模态，每个模态单独标准化
            normalized = np.zeros_like(data)
            for m in range(data.shape[3]):
                if method == 'z-score':
                    mean_val = np.mean(data[:, :, :, m])
                    std_val = np.std(data[:, :, :, m])
                    normalized[:, :, :, m] = (data[:, :, :, m] - mean_val) / (std_val + 1e-8)
                elif method == 'min-max':
                    min_val = np.min(data[:, :, :, m])
                    max_val = np.max(data[:, :, :, m])
                    normalized[:, :, :, m] = (data[:, :, :, m] - min_val) / (max_val - min_val + 1e-8)
        else:  # 单模态
            if method == 'z-score':
                mean_val = np.mean(data)
                std_val = np.std(data)
                normalized = (data - mean_val) / (std_val + 1e-8)
            elif method == 'min-max':
                min_val = np.min(data)
                max_val = np.max(data)
                normalized = (data - min_val) / (max_val - min_val + 1e-8)
        
        return normalized
    
    def process_label(self, label):
        """处理标签：转换为整数，二值化（0 vs 非0）"""
        # 转换为整数
        label = label.astype(np.int64)
        # 二值化：0保持为0，非0全部设为1
        label = (label > 0).astype(np.int64)
        return label
    
    def validate_pair(self, image, label):
        """验证图像和标签尺寸是否一致"""
        if len(image.shape) == 4:  # 多模态
            return image.shape[:3] == label.shape
        else:
            return image.shape == label.shape


class DataAugmenter:
    """数据增强类（仅用于训练集）"""
    
    def __init__(self, enable=True):
        self.enable = enable
    
    def random_flip(self, image, label):
        """随机翻转"""
        if not self.enable:
            return image, label
        
        # 随机沿三个轴翻转
        if random.random() > 0.5:
            if len(image.shape) == 4:
                image = np.flip(image, axis=0).copy()
            else:
                image = np.flip(image, axis=0).copy()
            label = np.flip(label, axis=0).copy()
        
        if random.random() > 0.5:
            if len(image.shape) == 4:
                image = np.flip(image, axis=1).copy()
            else:
                image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()
        
        if random.random() > 0.5:
            if len(image.shape) == 4:
                image = np.flip(image, axis=2).copy()
            else:
                image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=2).copy()
        
        return image, label
    
    def random_rotation(self, image, label, max_angle=15):
        """随机小角度旋转"""
        if not self.enable:
            return image, label
        
        # 简化版：实际项目中可使用更专业的3D旋转
        angle = random.uniform(-max_angle, max_angle)
        # 这里简化为不实现复杂旋转，实际使用时可调用scipy.ndimage.rotate
        return image, label
    
    def add_gaussian_noise(self, image, mean=0, std=0.01):
        """添加高斯噪声（仅图像）"""
        if not self.enable:
            return image
        
        noise = np.random.normal(mean, std, image.shape)
        image = image + noise
        return image


class BraTSDataset(Dataset):
    """BraTS数据集加载类"""
    
    def __init__(self, data_dir, split='train', preprocess_config=None, augment=False):
        """
        Args:
            data_dir: 数据根目录 (包含train/val/test子目录)
            split: 'train', 'val', 'test'
            preprocess_config: 预处理配置字典
            augment: 是否启用数据增强（仅对train有效）
        """
        self.data_dir = data_dir
        self.split = split
        self.preprocess_config = preprocess_config if preprocess_config else {}
        self.augment = augment and (split == 'train')
        
        # 获取文件路径列表
        self.image_dir = os.path.join(data_dir, split, 'image')
        self.label_dir = os.path.join(data_dir, split, 'labels')
        
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.nii')])
        self.label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.nii')])
        
        # 校验配对
        self._validate_files()
        
        # 初始化处理器和增强器
        self.processor = MedicalImageProcessor(
            target_spacing=self.preprocess_config.get('target_spacing', (1.0, 1.0, 1.0)),
            target_size=self.preprocess_config.get('target_size', (128, 128, 128))
        )
        self.augmenter = DataAugmenter(enable=self.augment)
    
    def _validate_files(self):
        """校验图像和标签文件配对"""
        image_names = set([f.split('.')[0] for f in self.image_files])
        label_names = set([f.split('.')[0] for f in self.label_files])
        
        if image_names != label_names:
            missing_in_labels = image_names - label_names
            missing_in_images = label_names - image_names
            raise ValueError(f"文件配对错误！\n"
                           f"图像中缺少对应标签: {missing_in_labels}\n"
                           f"标签中缺少对应图像: {missing_in_images}")
        
        print(f"[{self.split}] 文件配对校验通过，共 {len(self.image_files)} 个样本")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载数据
        image_file = os.path.join(self.image_dir, self.image_files[idx])
        label_file = os.path.join(self.label_dir, self.label_files[idx])
        
        # 读取NIfTI
        image_data, image_spacing, _ = self.processor.load_nifti(image_file)
        label_data, label_spacing, _ = self.processor.load_nifti(label_file)
        
        # 预处理
        # 1. 重采样（如果需要）
        if image_spacing[:3] != self.processor.target_spacing:
            image_data = self.processor.resample_to_target_spacing(
                image_data, image_spacing[:3], self.processor.target_spacing, order=3
            )
            label_data = self.processor.resample_to_target_spacing(
                label_data, label_spacing[:3], self.processor.target_spacing, order=0  # 最近邻
            )
        
        # 2. 调整尺寸
        image_data = self.processor.resize_to_target(image_data, self.processor.target_size)
        label_data = self.processor.resize_to_target(label_data, self.processor.target_size)
        
        # 3. 图像强度标准化
        image_data = self.processor.normalize_intensity(
            image_data, method=self.preprocess_config.get('norm_method', 'z-score')
        )
        
        # 4. 标签处理
        label_data = self.processor.process_label(label_data)
        
        # 5. 验证尺寸
        if len(image_data.shape) == 4:
            assert image_data.shape[:3] == label_data.shape, "图像和标签尺寸不一致！"
        else:
            assert image_data.shape == label_data.shape, "图像和标签尺寸不一致！"
        
        # 数据增强（仅训练集）
        if self.augment:
            image_data, label_data = self.augmenter.random_flip(image_data, label_data)
            image_data = self.augmenter.add_gaussian_noise(image_data)
        
        # 转换为PyTorch格式
        # NIfTI: (H, W, D) 或 (H, W, D, C)
        # PyTorch: (C, D, H, W)
        if len(image_data.shape) == 4:  # 多模态 (H, W, D, C)
            image_data = np.transpose(image_data, (3, 2, 0, 1))  # -> (C, D, H, W)
        else:  # 单模态 (H, W, D)
            image_data = np.expand_dims(image_data, axis=0)  # -> (1, D, H, W)
            image_data = np.transpose(image_data, (0, 3, 1, 2))  # -> (1, D, H, W)
        
        # 标签 (H, W, D) -> (D, H, W)
        label_data = np.transpose(label_data, (2, 0, 1))
        
        # 转换为tensor
        image_tensor = torch.from_numpy(image_data).float()
        label_tensor = torch.from_numpy(label_data).long()
        
        return {
            'image': image_tensor,
            'label': label_tensor,
            'filename': self.image_files[idx]
        }


def create_data_loaders(data_dir, preprocess_config, batch_size=2, num_workers=4):
    """创建训练、验证、测试数据加载器"""
    
    # 创建Dataset
    train_dataset = BraTSDataset(data_dir, split='train', 
                                  preprocess_config=preprocess_config, augment=True)
    val_dataset = BraTSDataset(data_dir, split='val', 
                                preprocess_config=preprocess_config, augment=False)
    test_dataset = BraTSDataset(data_dir, split='test', 
                                 preprocess_config=preprocess_config, augment=False)
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers, 
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, 
                            pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=num_workers, 
                             pin_memory=True, drop_last=False)
    
    return train_loader, val_loader, test_loader


# 测试代码
if __name__ == '__main__':
    # 测试数据加载
    data_dir = r'D:\PythonProgram\BrainSeg'
    preprocess_config = {
        'target_spacing': (1.0, 1.0, 1.0),
        'target_size': (128, 128, 128),
        'norm_method': 'z-score'
    }
    
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir, preprocess_config, batch_size=2, num_workers=0
    )
    
    # 测试一个batch
    for batch in train_loader:
        print(f"图像形状: {batch['image'].shape}")
        print(f"标签形状: {batch['label'].shape}")
        print(f"文件名: {batch['filename']}")
        break
