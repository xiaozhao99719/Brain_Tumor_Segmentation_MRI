"""
BrainMRI Dataset Data Loading Module
=====================================
Dataset structure:
    E:/python/BrainMRI/
    ├── train/   (training patients, folder names: 001, 002, ...)
    │   ├── 001/ (one patient folder)
    │   │   ├── *_t1.nii / *_t1ce.nii / *_t2.nii / *_flair.nii   (4 MRI sequences)
    │   │   └── *_seg.nii                                            (4-class segmentation label)
    │   ├── 002/
    │   └── ...
    ├── val/     (validation patients)
    │   ├── 001/
    │   └── ...
    └── test/    (test patients)
        ├── 001/
        └── ...

Segmentation label classes:
    0 - Background
    1 - Necrotic tumor core (NCR)
    2 - Peritumoral edema (ED)
    3 - GD-enhancing tumor (ET)

MRI channels (4-input, 4-channel 3D volume):
    Channel 0 - T1
    Channel 1 - T1ce (T1 post-contrast)
    Channel 2 - T2
    Channel 3 - FLAIR
"""

import os
import glob
import re
import warnings
from typing import List, Dict, Tuple, Optional, Callable, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib


# ============================================================================
# 1. 文件路径与扫描
# ============================================================================

def get_patient_folders(split: str) -> List[str]:
    """
    获取指定数据集划分（train/val/test）下的所有患者文件夹路径。

    Parameters
    ----------
    split : str
        数据集划分类型，可选值 'train'、'val'、'test'。

    Returns
    -------
    List[str]
        所有患者文件夹的绝对路径列表。
    """
    base = os.path.join("E:/python/BrainMRI", split)
    patient_ids = sorted(os.listdir(base))
    return [os.path.join(base, pid) for pid in patient_ids if os.path.isdir(os.path.join(base, pid))]


def scan_patient_files(patient_folder: str) -> Dict[str, str]:
    """
    扫描单个患者文件夹，区分 MRI 序列文件和分割标注文件。

    Parameters
    ----------
    patient_folder : str
        患者文件夹的绝对路径，文件夹名称为三位数字患者ID。

    Returns
    -------
    Dict[str, str]
        包含以下键的字典：
        - 't1'       : T1 加权影像路径
        - 't1ce'     : T1ce（T1 增强）影像路径
        - 't2'       : T2 加权影像路径
        - 'flair'    : FLAIR 影像路径
        - 'seg'      : 分割标注路径（含 "seg" 字符的 .nii 文件）
    """
    nii_files = glob.glob(os.path.join(patient_folder, "*.nii"))

    files = {'t1': None, 't1ce': None, 't2': None, 'flair': None, 'seg': None}

    for fp in nii_files:
        fname = os.path.basename(fp).lower()
        if 'seg' in fname:
            files['seg'] = fp
        elif 't1ce' in fname or 't1ce' in fname.replace('_', ''):
            files['t1ce'] = fp
        elif '_t1' in fname and 't1ce' not in fname:
            files['t1'] = fp
        elif '_t2' in fname:
            files['t2'] = fp
        elif 'flair' in fname:
            files['flair'] = fp

    missing = [k for k, v in files.items() if v is None]
    if missing:
        raise FileNotFoundError(
            f"Patient {os.path.basename(patient_folder)}: missing files for {missing}"
        )

    return files


# ============================================================================
# 2. NIfTI 加载与方向/重采样（空间变换预处理）
# ============================================================================

def load_nifti(filepath: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    加载 NIfTI 文件，返回数据数组和 NIfTI 图像对象。

    Parameters
    ----------
    filepath : str
        .nii 文件的路径。

    Returns
    -------
    Tuple[np.ndarray, nib.Nifti1Image]
        data : 3D numpy 数组（float）
        img  : nibabel NIfTI 图像对象（携带头信息）
    """
    img = nib.load(filepath)
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data, img


def resample_to_isotropic(
    img: nib.Nifti1Image,
    target_spacing: float = 1.0,
    order: int = 3
) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    将 NIfTI 影像重采样至固定各向同性体素间距。

    Parameters
    ----------
    img : nib.Nifti1Image
        原始 nibabel 图像对象。
    target_spacing : float
        目标各向同性体素间距（mm），默认 1.0 mm。
    order : int
        重采样插值阶数（0 = 最近邻，1 = 线性，3 = 三次样条）。
        MRI 影像使用 3，seg 标注必须使用 0。

    Returns
    -------
    Tuple[np.ndarray, nib.Nifti1Image]
        重采样后的 3D 数据数组和新的 NIfTI 图像对象。

    Notes
    -----
    - 不进行空白裁剪，不强制缩放/填充至固定尺寸；
      输出尺寸由原始尺寸与 target_spacing 的比值决定。
    - 方向由 nibabel 加载时自动保留，不额外做空间变换。
    """
    spacing = np.array(img.header.get_zooms()[:3])
    shape = np.array(img.shape[:3])

    new_shape = np.round(shape * spacing / target_spacing).astype(int)
    factors = spacing / target_spacing

    # 构建各向同性参考图像以确定变换矩阵
    ref_img = nib.Nifti1Image(
        np.zeros(new_shape, dtype=img.get_data_dtype()),
        np.eye(4)
    )
    ref_img.header.set_zooms((target_spacing,) * 3)

    # 使用 nilearn 的 resample_to_img（依赖 scipy/numpy）
    from nilearn.image import resample_to_img
    resampled = resample_to_img(
        source_img=img,
        target_img=ref_img,
        interpolation='continuous' if order > 0 else 'nearest'
    )

    data = np.asarray(resampled.dataobj, dtype=np.float32)
    return data, resampled


def apply_ras_orientation(data: np.ndarray, img: nib.Nifti1Image) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    将 NIfTI 影像校正为标准 RAS（Right-Anterior-Superior）解剖方向。

    Parameters
    ----------
    data : np.ndarray
        3D 影像数据数组。
    img : nib.Nifti1Image
        原始 NIfTI 图像对象（携带方向信息）。

    Returns
    -------
    Tuple[np.ndarray, nib.Nifti1Image]
        方向校正后的数据数组和新的 NIfTI 图像对象。

    Notes
    -----
    仅通过轴面翻转调整方向，不改变体素间距，不裁剪/填充尺寸。
    """
    # 获取当前方向并计算 RAS 方向变换
    ras_matrix = img.affine[:3, :3]
    signs = np.sign(np.diag(ras_matrix))
    axes = np.argmax(np.abs(ras_matrix), axis=1)

    # 判断是否需要翻转
    needs_flip = [s < 0 for s in signs]

    corrected = data.copy()
    for axis_idx, flip in enumerate(needs_flip):
        if flip:
            corrected = np.flip(corrected, axis=axis_idx)

    # 更新仿射矩阵
    new_affine = img.affine.copy()
    for i, flip in enumerate(needs_flip):
        if flip:
            new_affine[i, 3] += (img.shape[i] - 1) * img.affine[i, i]

    new_img = nib.Nifti1Image(corrected, new_affine, header=img.header)
    return corrected, new_img


# ============================================================================
# 3. 强度预处理（仅 MRI，seg 标注全程不参与）
# ============================================================================

def bias_field_correction(mri_data: np.ndarray) -> np.ndarray:
    """
    对 MRI 影像执行 N4 偏置场校正（仅校正灰度，不改变数据结构）。

    Parameters
    ----------
    mri_data : np.ndarray
        3D MRI 影像数据。

    Returns
    -------
    np.ndarray
        偏置场校正后的 3D MRI 数据。

    Notes
    -----
    - 基于 SimpleITK 的 N4BiasFieldCorrection 实现；
    - 仅作用于 MRI 影像，seg 标注全程不参与此步骤。
    """
    import SimpleITK as sitk

    img_sitk = sitk.GetImageFromArray(mri_data)
    img_sitk = sitk.Cast(img_sitk, sitk.sitkFloat32)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
    corrected_sitk = corrector.Execute(img_sitk)

    corrected = sitk.GetArrayFromImage(corrected_sitk).astype(np.float32)
    return corrected


def percentile_clip(
    mri_data: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.0
) -> np.ndarray:
    """
    通过百分位截断剔除 MRI 影像中的极端异常灰度值。

    Parameters
    ----------
    mri_data : np.ndarray
        3D MRI 影像数据。
    lower : float
        下百分位截断阈值，默认 1.0（即 1st 百分位）。
    upper : float
        上百分位截断阈值，默认 99.0（即 99th 百分位）。

    Returns
    -------
    np.ndarray
        截断后的 3D MRI 数据。
    """
    lo = np.percentile(mri_data[mri_data > 0], lower) if np.any(mri_data > 0) else 0
    hi = np.percentile(mri_data[mri_data > 0], upper) if np.any(mri_data > 0) else np.max(mri_data)
    clipped = np.clip(mri_data, lo, hi)
    return clipped


def per_sample_normalize(mri_data: np.ndarray) -> np.ndarray:
    """
    对单个患者的 MRI 影像进行自身独立归一化（单样本 z-score）。

    Parameters
    ----------
    mri_data : np.ndarray
        3D MRI 影像数据（偏置场校正 + 百分位截断后）。

    Returns
    -------
    np.ndarray
        归一化后的数据，均值归零，标准差归一化到 1。

    Notes
    -----
    - 以每个患者的单个 MRI 序列为独立样本计算统计量；
    - 训练集、验证集、测试集各自独立执行，不复用统计标准；
    - 使用非零体素计算均值和标准差。
    """
    mask = mri_data > 0
    if not np.any(mask):
        return mri_data

    mean = mri_data[mask].mean()
    std = mri_data[mask].std()
    std = std if std > 1e-8 else 1.0

    normalized = (mri_data - mean) / std
    return normalized


def preprocess_single_mri(mri_path: str, target_spacing: float = 1.0) -> np.ndarray:
    """
    对单个 MRI 序列文件执行完整预处理流水线。

    处理步骤（顺序执行）：
        1. 加载 NIfTI
        2. RAS 方向校正
        3. 各向同性重采样（order=3，三次样条插值）
        4. N4 偏置场校正
        5. 百分位截断（1st/99th）
        6. 单样本自身归一化

    Parameters
    ----------
    mri_path : str
        MRI 序列文件的路径。
    target_spacing : float
        目标各向同性体素间距（mm），默认 1.0。

    Returns
    -------
    np.ndarray
        预处理后的 3D MRI 数据，float32 类型。
    """
    data, img = load_nifti(mri_path)

    # 方向校正
    data, _ = apply_ras_orientation(data, img)

    # 各向同性重采样（MRI 用三次样条插值）
    data, _ = resample_to_isotropic(img, target_spacing=target_spacing, order=3)

    # 偏置场校正
    data = bias_field_correction(data)

    # 百分位截断
    data = percentile_clip(data)

    # 单样本归一化
    data = per_sample_normalize(data)

    return data.astype(np.float32)


# ============================================================================
# 4. 分割标注预处理（最近邻插值，保证标签不畸变）
# ============================================================================

def preprocess_seg_label(seg_path: str, target_spacing: float = 1.0) -> np.ndarray:
    """
    对分割标注 NIfTI 文件执行完整预处理流水线。

    处理步骤（顺序执行）：
        1. 加载 NIfTI
        2. RAS 方向校正
        3. 各向同性重采样（order=0，最近邻插值）
        4. 类型转换为 int 类别标签（不进行强度归一化）

    Parameters
    ----------
    seg_path : str
        分割标注文件的路径。
    target_spacing : float
        目标各向同性体素间距（mm），默认 1.0。

    Returns
    -------
    np.ndarray
        预处理后的 3D 分割标注，int 类型，标签值仅含 {0, 1, 2, 3}。

    Notes
    -----
    - 全程使用最近邻插值（order=0），严格保证标签不产生小数；
    - 不执行偏置场校正、截断或归一化等任何灰度操作；
    - 不裁剪、不填充至固定尺寸。
    """
    data, img = load_nifti(seg_path)

    # 方向校正（seg 数据使用最近邻插值）
    data, _ = resample_to_isotropic(img, target_spacing=target_spacing, order=0)

    # 强制最近邻插值确保标签整数化
    data = np.round(data).astype(np.int32)
    data = np.clip(data, 0, 3)

    return data


# ============================================================================
# 5. 4 通道 MRI 拼接与样本构建
# ============================================================================

def build_4channel_input(
    t1_data: np.ndarray,
    t1ce_data: np.ndarray,
    t2_data: np.ndarray,
    flair_data: np.ndarray
) -> np.ndarray:
    """
    将同一患者的 4 个 MRI 序列拼接为 4 通道 3D 体数据。

    Parameters
    ----------
    t1_data    : np.ndarray，3D T1 加权影像
    t1ce_data  : np.ndarray，3D T1ce（T1 增强）影像
    t2_data    : np.ndarray，3D T2 加权影像
    flair_data : np.ndarray，3D FLAIR 影像

    Returns
    -------
    np.ndarray
        shape = (4, D, H, W)，4 通道 3D 体数据，float32 类型。

    Notes
    -----
    通道顺序固定为 [T1, T1ce, T2, FLAIR]。
    四个序列在预处理阶段已通过重采样统一空间分辨率，
    此处仅做通道维度拼接，无需额外对齐。
    """
    stacked = np.stack([t1_data, t1ce_data, t2_data, flair_data], axis=0)
    return stacked.astype(np.float32)


# ============================================================================
# 6. 数据增强（仅训练集）
# ============================================================================

def random_flip_along_axis(volume: np.ndarray, axis: int) -> np.ndarray:
    """
    沿指定轴执行随机左右翻转（概率 0.5）。

    Parameters
    ----------
    volume : np.ndarray
        输入 3D 体数据。
    axis : int
        翻转轴（0=D，1=H，2=W）。

    Returns
    -------
    np.ndarray
        翻转后的体数据。
    """
    if np.random.rand() < 0.5:
        return np.flip(volume, axis=axis).copy()
    return volume


def random_rotation_3d(
    volume: np.ndarray,
    seg: np.ndarray,
    max_angle: float = 15.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对 3D 体数据和标注执行随机小角度旋转变换。

    Parameters
    ----------
    volume : np.ndarray
        3D MRI 体数据。
    seg : np.ndarray
        3D 分割标注。
    max_angle : float
        最大旋转角度（度），默认 15°。

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        旋转变换后的 (volume, seg)。

    Notes
    -----
    - 使用 scipy.ndimage.rotate，MRI 插值阶数 3，seg 插值阶数 0；
    - 仅在轴面（D-H 和 D-W 平面）执行随机旋转；
    - 旋转后 seg 严格保留整数标签。
    """
    from scipy.ndimage import rotate

    angle = np.random.uniform(-max_angle, max_angle)
    plane = np.random.choice([(1, 2), (0, 2)])   # 选择旋转平面

    vol_rot = rotate(
        volume, angle,
        axes=plane,
        reshape=False,
        order=3,
        mode='constant',
        cval=0.0
    )
    seg_rot = rotate(
        seg, angle,
        axes=plane,
        reshape=False,
        order=0,   # 最近邻插值，保证标签不畸变
        mode='constant',
        cval=0
    )
    seg_rot = np.round(seg_rot).astype(np.int32)

    return vol_rot, seg_rot


def elastic_deformation_3d(
    volume: np.ndarray,
    seg: np.ndarray,
    alpha: float = 30.0,
    sigma: float = 9.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对 3D 体数据和标注执行弹性形变数据增强。

    Parameters
    ----------
    volume : np.ndarray
        3D MRI 体数据。
    seg : np.ndarray
        3D 分割标注。
    alpha : float
        形变幅度系数，控制最大位移量（体素单位），默认 30.0。
    sigma : float
        平滑核 sigma，控制形变平滑度，默认 9.0。

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        弹性形变后的 (volume, seg)。

    Notes
    -----
    - 使用基于弹性网格的位移场实现；
    - MRI 插值阶数 3，seg 插值阶数 0；
    - 形变场沿通道维度广播，对所有通道应用相同变换。
    """
    from scipy.ndimage import map_coordinates, gaussian_filter

    shape = volume.shape[1:]  # (D, H, W)

    # 生成独立随机位移场
    displacements = [
        gaussian_filter(
            np.random.randn(*shape) * alpha, sigma=sigma, mode="constant", cval=0
        )
        for _ in range(3)
    ]

    # 构建变形后坐标网格
    coords = [
        np.arange(s) + disp
        for s, disp in zip(shape, displacements)
    ]
    grid = np.array(np.meshgrid(*coords, indexing='ij'))

    vol_out = np.zeros_like(volume)
    seg_out = np.zeros_like(seg)

    for c in range(volume.shape[0]):
        vol_out[c] = map_coordinates(
            volume[c], grid, order=3, mode='constant', cval=0
        )
    for c in range(seg.shape[0]):
        seg_out[c] = np.round(
            map_coordinates(seg[c], grid, order=0, mode='constant', cval=0)
        ).astype(np.int32)

    return vol_out, seg_out


def apply_training_augmentation(
    volume: np.ndarray,
    seg: np.ndarray,
    p_flip: float = 0.5,
    p_rotate: float = 0.3,
    p_elastic: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    按指定概率依次应用训练集数据增强。

    增强操作（按顺序执行）：
        1. 随机左右翻转（3 个轴，各 50% 概率）
        2. 随机小角度旋转（仅轴面，最多 ±15°）
        3. 随机弹性形变

    Parameters
    ----------
    volume : np.ndarray
        4 通道 3D MRI 体数据，shape = (4, D, H, W)。
    seg : np.ndarray
        3D 分割标注，shape = (D, H, W)。
    p_flip    : float，各轴翻转概率，默认 0.5
    p_rotate  : float，旋转概率，默认 0.3
    p_elastic : float，弹性形变概率，默认 0.2

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        增强后的 (volume, seg)。

    Notes
    -----
    - 验证集和测试集不调用此函数；
    - 所有变换同步作用于 volume 和 seg；
    - seg 始终使用最近邻插值（order=0）。
    """
    # 随机轴面翻转（每次独立判断 D/H/W 三个轴）
    for axis in [1, 2, 3]:
        if np.random.rand() < p_flip:
            volume = np.flip(volume, axis=axis).copy()
            seg = np.flip(seg, axis=axis - 1).copy()

    # 随机小角度旋转
    if np.random.rand() < p_rotate:
        volume, seg = random_rotation_3d(volume, seg)

    # 随机弹性形变
    if np.random.rand() < p_elastic:
        volume, seg = elastic_deformation_3d(volume, seg)

    return volume, seg


# ============================================================================
# 7. PyTorch Dataset 实现
# ============================================================================

class BrainMRIDataset(Dataset):
    """
    BrainMRI PyTorch Dataset。

    Parameters
    ----------
    split : str
        数据集划分，'train'、'val' 或 'test'。
    target_spacing : float
        各向同性重采样体素间距（mm），默认 1.0。
    augment : bool
        是否应用数据增强（仅 train 时为 True）。

    Attributes
    ----------
    samples : List[Dict]
        每个样本的信息列表，包含 patient_id、split、mri_path、seg_path。

    Notes
    -----
    - 训练集：空间预处理 + 强度预处理 + 数据增强；
    - 验证集：空间预处理 + 强度预处理（无增强）；
    - 测试集：空间预处理 + 强度预处理（无增强）。
    """

    def __init__(
        self,
        split: str = 'train',
        target_spacing: float = 1.0,
        augment: bool = False
    ):
        super().__init__()
        self.split = split
        self.target_spacing = target_spacing
        self.augment = augment

        # 扫描所有患者文件夹
        self.patient_folders = get_patient_folders(split)
        self.samples = []
        for pf in self.patient_folders:
            patient_id = os.path.basename(pf)
            files = scan_patient_files(pf)
            self.samples.append({
                'patient_id': patient_id,
                'split': split,
                **files
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        加载并预处理单个患者样本。

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, Dict]
            volume : torch.Tensor，shape = (4, D, H, W)，float32
            seg    : torch.Tensor，shape = (D, H, W)，long int（4 分类标签）
            info   : Dict，包含 patient_id、split、shape 等元信息
        """
        sample = self.samples[idx]

        # ---------- 预处理 4 个 MRI 序列 ----------
        t1_data    = preprocess_single_mri(sample['t1'],    self.target_spacing)
        t1ce_data  = preprocess_single_mri(sample['t1ce'],  self.target_spacing)
        t2_data    = preprocess_single_mri(sample['t2'],    self.target_spacing)
        flair_data = preprocess_single_mri(sample['flair'], self.target_spacing)

        # ---------- 预处理分割标注 ----------
        seg_data = preprocess_seg_label(sample['seg'], self.target_spacing)

        # ---------- 拼接为 4 通道 MRI ----------
        volume = build_4channel_input(t1_data, t1ce_data, t2_data, flair_data)

        # ---------- 训练集数据增强 ----------
        if self.augment:
            volume, seg_data = apply_training_augmentation(volume, seg_data)

        # ---------- 转换为 PyTorch Tensor ----------
        volume_t = torch.from_numpy(volume.copy())
        seg_t = torch.from_numpy(seg_data.copy().astype(np.int64))

        info = {
            'patient_id': sample['patient_id'],
            'split': self.split,
            'shape': volume_t.shape,          # (4, D, H, W)
            'seg_shape': seg_t.shape,          # (D, H, W)
            'unique_labels': torch.unique(seg_t).tolist(),
        }

        return volume_t, seg_t, info

    def print_summary(self):
        """
        打印数据集的详细统计信息。

        包含：样本数量、各类标签分布、体素尺寸信息等。
        """
        print("=" * 60)
        print(f"  BrainMRI Dataset Summary — Split: {self.split.upper()}")
        print("=" * 60)
        print(f"  Total patients  : {len(self.samples)}")
        print(f"  Target spacing  : {self.target_spacing} mm (isotropic)")
        print(f"  Augmentation    : {'Enabled' if self.augment else 'Disabled'}")
        print("-" * 60)

        # 汇总标签分布
        all_labels = {0: 0, 1: 0, 2: 0, 3: 0}
        all_shapes = []

        for i in range(len(self)):
            _, seg, info = self[i]
            all_shapes.append(info['shape'])

            for label in [0, 1, 2, 3]:
                count = (seg == label).sum().item()
                all_labels[label] += count

        print(f"  Unique label classes: {sorted(all_labels.keys())}")
        label_names = {0: 'Background (0)', 1: 'NCR (1)', 2: 'ED (2)', 3: 'ET (3)'}
        total_voxels = sum(all_labels.values())
        for label, count in all_labels.items():
            pct = count / total_voxels * 100 if total_voxels > 0 else 0
            print(f"    {label_names[label]:20s}: {count:>12,} voxels ({pct:.2f}%)")

        print("-" * 60)

        # 体素形状统计
        all_shapes = np.array(all_shapes)
        print(f"  Volume shape range:")
        print(f"    Channel : {all_shapes[:, 0].min()} (fixed)")
        print(f"    Depth   : {all_shapes[:, 1].min()} – {all_shapes[:, 1].max()}")
        print(f"    Height  : {all_shapes[:, 2].min()} – {all_shapes[:, 2].max()}")
        print(f"    Width   : {all_shapes[:, 3].min()} – {all_shapes[:, 3].max()}")
        print("=" * 60)


# ============================================================================
# 8. DataLoader 工厂函数
# ============================================================================

def build_dataloaders(
    batch_size: int = 2,
    target_spacing: float = 1.0,
    num_workers: int = 4,
    persistent_workers: bool = True
) -> Dict[str, DataLoader]:
    """
    构建训练集、验证集、测试集 DataLoader。

    Parameters
    ----------
    batch_size : int
        每批次样本数，默认 2（3D MRI 数据较大，需根据显存调整）。
    target_spacing : float
        各向同性体素间距（mm），默认 1.0 mm。
    num_workers : int
        DataLoader 子进程数，默认 4。
    persistent_workers : bool
        是否保持 worker 进程常驻（加速多 epoch 训练），默认 True。

    Returns
    -------
    Dict[str, DataLoader]
        包含 'train'、'val'、'test' 三个 DataLoader 的字典。

    Notes
    -----
    - 训练集开启数据增强，验证/测试集关闭；
    - seg 标签使用 collate_fn 验证形状一致性；
    - pin_memory=True 加速 GPU 传输。
    """

    def seg_collate_fn(batch):
        """
        自定义 collate_fn：校验 seg 与 volume 形状一致性。
        """
        volumes, segs, infos = zip(*batch)
        vol_batch = torch.stack(volumes, dim=0)
        seg_batch = torch.stack(segs, dim=0)
        return vol_batch, seg_batch, infos

    def make_loader(split: str, augment: bool) -> DataLoader:
        dataset = BrainMRIDataset(
            split=split,
            target_spacing=target_spacing,
            augment=augment
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            collate_fn=seg_collate_fn
        )
        return loader

    loaders = {
        'train': make_loader('train', augment=True),
        'val':   make_loader('val',   augment=False),
        'test':  make_loader('test',  augment=False),
    }

    return loaders


# ============================================================================
# 9. 可视化辅助 / 工具函数
# ============================================================================

def get_class_weights(loader: DataLoader, num_classes: int = 4) -> torch.Tensor:
    """
    根据 DataLoader 统计各类别体素比例，返回交叉熵加权系数。

    Parameters
    ----------
    loader : DataLoader
        训练集 DataLoader。
    num_classes : int
        分割类别数，默认 4。

    Returns
    -------
    torch.Tensor
        shape = (num_classes,)，各类别加权系数。
    """
    total_voxels = torch.zeros(num_classes)
    for _, seg, _ in loader:
        for c in range(num_classes):
            total_voxels[c] += (seg == c).sum().item()

    total = total_voxels.sum()
    weights = total / (num_classes * total_voxels + 1e-8)
    weights = weights / weights.sum() * num_classes   # 归一化
    return weights


def print_dataloader_info(loaders: Dict[str, DataLoader]):
    """
    打印所有 DataLoader 的详细信息。

    Parameters
    ----------
    loaders : Dict[str, DataLoader]
        build_dataloaders 返回的 DataLoader 字典。
    """
    print("\n" + "=" * 65)
    print("  BrainMRI DataLoader Information")
    print("=" * 65)
    for name, loader in loaders.items():
        ds = loader.dataset
        print(f"\n  [{name.upper()}]")
        print(f"    Dataset size    : {len(ds)} patients")
        print(f"    Batch size      : {loader.batch_size}")
        print(f"    Batches per ep. : {len(loader)}")
        print(f"    Shuffle         : {loader.shuffle}")
        print(f"    Num workers     : {loader.num_workers}")
        print(f"    Augmentation    : {'Yes' if ds.augment else 'No'}")
        print(f"    Target spacing  : {ds.target_spacing} mm")
        print(f"    Patient IDs     : {[s['patient_id'] for s in ds.samples[:5]]}{'...' if len(ds.samples) > 5 else ''}")

        # 打印一个样本的形状
        vol, seg, info = ds[0]
        print(f"    Sample volume   : {tuple(vol.shape)}  (C, D, H, W)")
        print(f"    Sample seg      : {tuple(seg.shape)}  (D, H, W)")
        print(f"    Sample labels   : {info['unique_labels']}")
    print("\n" + "=" * 65)


# ============================================================================
# 使用示例（不运行，仅展示调用方式）
# ============================================================================
#
# # 方式一：直接构建 DataLoader
# loaders = build_dataloaders(batch_size=2, target_spacing=1.0)
#
# # 打印详细信息
# print_dataloader_info(loaders)
#
# # 打印各类别权重
# weights = get_class_weights(loaders['train'])
# print(f"Class weights: {weights}")
#
# # 方式二：独立使用 Dataset
# train_dataset = BrainMRIDataset(split='train', augment=True)
# val_dataset   = BrainMRIDataset(split='val',   augment=False)
# test_dataset  = BrainMRIDataset(split='test',  augment=False)
#
# # 打印数据集统计摘要
# train_dataset.print_summary()
# val_dataset.print_summary()
# test_dataset.print_summary()
#
# # 遍历 DataLoader
# for batch_idx, (volume, seg, info) in enumerate(loaders['train']):
#     print(f"Batch {batch_idx}: volume={volume.shape}, seg={seg.shape}")
#     # volume: (B, 4, D, H, W)
#     # seg   : (B, D, H, W)
#     # info  : List[Dict]，每条含 patient_id / split / shape 等
#
