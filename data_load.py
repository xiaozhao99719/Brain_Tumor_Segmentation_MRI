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

Note: This module is for data preparation only, no training involved.
      CUDA acceleration is used where possible.
"""

import os
import glob
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset


# ============================================================================
# CUDA Configuration
# ============================================================================

def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, else CPU).
    
    Returns
    -------
    torch.device
        The device to use for computation.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Using device: CUDA (GPU)")
    else:
        device = torch.device('cpu')
        print("[INFO] CUDA not available, using CPU")
    return device


# Initialize device
DEVICE = get_device()


# ============================================================================
# 1. File Path and Scanning
# ============================================================================

def get_patient_folders(split: str) -> List[str]:
    """
    Get all patient folder paths for a given dataset split.

    Parameters
    ----------
    split : str
        Dataset split type, one of 'train', 'val', 'test'.

    Returns
    -------
    List[str]
        List of absolute paths to all patient folders.
    """
    base = os.path.join("E:/python/BrainMRI", split)
    patient_ids = sorted(os.listdir(base))
    return [os.path.join(base, pid) for pid in patient_ids if os.path.isdir(os.path.join(base, pid))]


def scan_patient_files(patient_folder: str) -> Dict[str, str]:
    """
    Scan a single patient folder to identify MRI sequence files and segmentation label files.

    Parameters
    ----------
    patient_folder : str
        Absolute path to the patient folder, folder name is a 3-digit patient ID.

    Returns
    -------
    Dict[str, str]
        Dictionary containing:
        - 't1'       : T1-weighted image path
        - 't1ce'     : T1ce (T1 contrast-enhanced) image path
        - 't2'       : T2-weighted image path
        - 'flair'    : FLAIR image path
        - 'seg'      : Segmentation label path (.nii file containing "seg")
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
# 2. NIfTI Loading and Orientation/Resampling (Spatial Preprocessing)
# ============================================================================

def load_nifti(filepath: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Load a NIfTI file, returning data array and NIfTI image object.

    Parameters
    ----------
    filepath : str
        Path to the .nii file.

    Returns
    -------
    Tuple[np.ndarray, nib.Nifti1Image]
        data : 3D numpy array (float)
        img  : nibabel NIfTI image object (carrying header info)
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
    Resample NIfTI image to fixed isotropic voxel spacing using CUDA-accelerated interpolation.

    Parameters
    ----------
    img : nib.Nifti1Image
        Original nibabel image object.
    target_spacing : float
        Target isotropic voxel spacing (mm), default 1.0 mm.
    order : int
        Resampling interpolation order (0 = nearest, 1 = linear, 3 = cubic).
        Use 3 for MRI images, must use 0 for seg labels.

    Returns
    -------
    Tuple[np.ndarray, nib.Nifti1Image]
        Resampled 3D data array and new NIfTI image object.

    Notes
    -----
    - Uses torch.nn.functional.interpolate for CUDA acceleration when available;
    - Output size is determined by original size / spacing ratio.
    """
    spacing = np.array(img.header.get_zooms()[:3])
    shape = np.array(img.shape[:3])

    zoom_factors = spacing / target_spacing
    new_shape = np.round(shape * zoom_factors).astype(int)

    data = img.get_fdata(dtype=np.float32)
    
    # Use CUDA-accelerated interpolation if available
    if torch.cuda.is_available() and order > 0:
        # Convert to torch tensor and use GPU interpolation
        data_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, D, H, W)
        
        # Calculate scale factors
        scale_factors = [new_shape[i] / shape[i] for i in range(3)]
        
        # Use trilinear/tricubic interpolation via upsampling
        # Note: torch.interpolate only supports trilinear (order=1), use scipy for cubic
        if order == 1:
            data_resampled = torch.nn.functional.interpolate(
                data_tensor,
                size=tuple(new_shape),
                mode='trilinear',
                align_corners=False
            )
            resampled_data = data_resampled.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        else:
            # Fallback to scipy for cubic interpolation
            from scipy.ndimage import zoom
            resampled_data = zoom(data, zoom_factors, order=order, mode='constant', cval=0.0)
    else:
        # Fallback to scipy.ndimage.zoom for CPU or nearest neighbor
        from scipy.ndimage import zoom
        resampled_data = zoom(data, zoom_factors, order=order, mode='constant', cval=0.0)

    # Update affine matrix to reflect new spacing
    zoom_diag = np.diag([
        1.0 / zoom_factors[0],
        1.0 / zoom_factors[1],
        1.0 / zoom_factors[2],
        1.0
    ])
    new_affine = img.affine @ zoom_diag

    new_img = nib.Nifti1Image(resampled_data, new_affine, header=img.header)

    return resampled_data.astype(np.float32), new_img


def apply_ras_orientation(data: np.ndarray, img: nib.Nifti1Image) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Correct NIfTI image to standard RAS (Right-Anterior-Superior) anatomical orientation.

    Parameters
    ----------
    data : np.ndarray
        3D image data array.
    img : nib.Nifti1Image
        Original NIfTI image object (carrying orientation info).

    Returns
    -------
    Tuple[np.ndarray, nib.Nifti1Image]
        Orientation-corrected data array and new NIfTI image object.

    Notes
    -----
    Only adjusts orientation via axis flipping, does not change voxel spacing or crop/pad.
    """
    ras_matrix = img.affine[:3, :3]
    signs = np.sign(np.diag(ras_matrix))

    corrected = data.copy()
    for axis_idx, s in enumerate(signs):
        if s < 0:
            corrected = np.flip(corrected, axis=axis_idx)

    # Update affine matrix
    new_affine = img.affine.copy()
    for i, s in enumerate(signs):
        if s < 0:
            new_affine[i, 3] += (img.shape[i] - 1) * img.affine[i, i]

    new_img = nib.Nifti1Image(corrected, new_affine, header=img.header)
    return corrected, new_img


# ============================================================================
# 3. Intensity Preprocessing (MRI only, seg labels not involved)
# ============================================================================

def bias_field_correction(mri_data: np.ndarray) -> np.ndarray:
    """
    Apply N4 bias field correction to MRI image (grayscale correction only).

    Parameters
    ----------
    mri_data : np.ndarray
        3D MRI image data.

    Returns
    -------
    np.ndarray
        Bias field corrected 3D MRI data.

    Notes
    -----
    - Based on SimpleITK's N4BiasFieldCorrection implementation.
    - Only applies to MRI images, seg labels are never processed here.
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
    Clip extreme intensity values in MRI image using percentile thresholds.

    Parameters
    ----------
    mri_data : np.ndarray
        3D MRI image data.
    lower : float
        Lower percentile threshold, default 1.0 (1st percentile).
    upper : float
        Upper percentile threshold, default 99.0 (99th percentile).

    Returns
    -------
    np.ndarray
        Clipped 3D MRI data.
    """
    lo = np.percentile(mri_data[mri_data > 0], lower) if np.any(mri_data > 0) else 0
    hi = np.percentile(mri_data[mri_data > 0], upper) if np.any(mri_data > 0) else np.max(mri_data)
    clipped = np.clip(mri_data, lo, hi)
    return clipped


def per_sample_normalize(mri_data: np.ndarray) -> np.ndarray:
    """
    Normalize single patient's MRI image using per-sample z-score.

    Parameters
    ----------
    mri_data : np.ndarray
        3D MRI image data (after bias field correction + percentile clipping).

    Returns
    -------
    np.ndarray
        Normalized data with zero mean and unit standard deviation.

    Notes
    -----
    - Uses non-zero voxels to compute mean and std.
    - Uses CUDA acceleration when available.
    """
    # Use CUDA for faster computation
    if torch.cuda.is_available():
        mri_tensor = torch.from_numpy(mri_data).to(DEVICE)
        mask = mri_tensor > 0
        if not mask.any():
            return mri_data
        
        mean = mri_tensor[mask].mean()
        std = mri_tensor[mask].std()
        std = std if std > 1e-8 else torch.tensor(1.0, device=DEVICE)
        
        normalized_tensor = (mri_tensor - mean) / std
        return normalized_tensor.cpu().numpy().astype(np.float32)
    else:
        mask = mri_data > 0
        if not np.any(mask):
            return mri_data
        
        mean = mri_data[mask].mean()
        std = mri_data[mask].std()
        std = std if std > 1e-8 else 1.0
        
        normalized = (mri_data - mean) / std
        return normalized.astype(np.float32)


def preprocess_single_mri(mri_path: str, target_spacing: float = 1.0) -> np.ndarray:
    """
    Execute complete preprocessing pipeline for a single MRI sequence file.

    Processing steps (in order):
        1. Load NIfTI
        2. RAS orientation correction
        3. Isotropic resampling (order=3, cubic interpolation)
        4. N4 bias field correction
        5. Percentile clipping (1st/99th)
        6. Per-sample normalization

    Parameters
    ----------
    mri_path : str
        Path to the MRI sequence file.
    target_spacing : float
        Target isotropic voxel spacing (mm), default 1.0.

    Returns
    -------
    np.ndarray
        Preprocessed 3D MRI data, float32 type.
    """
    data, img = load_nifti(mri_path)

    # Isotropic resampling (MRI uses cubic interpolation)
    data, _ = resample_to_isotropic(img, target_spacing=target_spacing, order=3)

    # RAS orientation correction
    data, _ = apply_ras_orientation(data, img)

    # Bias field correction
    data = bias_field_correction(data)

    # Percentile clipping
    data = percentile_clip(data)

    # Per-sample normalization
    data = per_sample_normalize(data)

    return data.astype(np.float32)


# ============================================================================
# 4. Segmentation Label Preprocessing (Nearest neighbor interpolation)
# ============================================================================

def preprocess_seg_label(seg_path: str, target_spacing: float = 1.0) -> np.ndarray:
    """
    Execute complete preprocessing pipeline for segmentation label NIfTI file.

    Processing steps (in order):
        1. Load NIfTI
        2. RAS orientation correction
        3. Isotropic resampling (order=0, nearest neighbor interpolation)
        4. Type conversion to int class labels (no intensity normalization)

    Parameters
    ----------
    seg_path : str
        Path to the segmentation label file.
    target_spacing : float
        Target isotropic voxel spacing (mm), default 1.0.

    Returns
    -------
    np.ndarray
        Preprocessed 3D segmentation label, int type, label values {0, 1, 2, 3}.

    Notes
    -----
    - Uses nearest neighbor interpolation (order=0) throughout to ensure no fractional labels.
    - No bias field correction, clipping, or normalization.
    """
    data, img = load_nifti(seg_path)

    # Isotropic resampling (seg uses nearest neighbor interpolation)
    data, _ = resample_to_isotropic(img, target_spacing=target_spacing, order=0)

    # RAS orientation correction
    data, _ = apply_ras_orientation(data, img)

    # Force nearest neighbor interpolation to ensure integer labels
    data = np.round(data).astype(np.int32)
    data = np.clip(data, 0, 3)

    return data


# ============================================================================
# 5. 4-Channel MRI Stacking and Sample Construction
# ============================================================================

def build_4channel_input(
    t1_data: np.ndarray,
    t1ce_data: np.ndarray,
    t2_data: np.ndarray,
    flair_data: np.ndarray
) -> np.ndarray:
    """
    Stack 4 MRI sequences of the same patient into a 4-channel 3D volume.

    Parameters
    ----------
    t1_data    : np.ndarray, 3D T1-weighted image
    t1ce_data  : np.ndarray, 3D T1ce (T1 contrast-enhanced) image
    t2_data    : np.ndarray, 3D T2-weighted image
    flair_data : np.ndarray, 3D FLAIR image

    Returns
    -------
    np.ndarray
        shape = (4, D, H, W), 4-channel 3D volume, float32 type.

    Notes
    -----
    Channel order is fixed: [T1, T1ce, T2, FLAIR].
    All sequences have been unified to the same spatial resolution via resampling.
    """
    stacked = np.stack([t1_data, t1ce_data, t2_data, flair_data], axis=0)
    return stacked.astype(np.float32)


# ============================================================================
# 6. PyTorch Dataset Implementation (Data Preparation Only)
# ============================================================================

class BrainMRIDataset(Dataset):
    """
    BrainMRI PyTorch Dataset for data preparation.

    Parameters
    ----------
    split : str
        Dataset split, 'train', 'val' or 'test'.
    target_spacing : float
        Isotropic resampling voxel spacing (mm), default 1.0.

    Attributes
    ----------
    samples : List[Dict]
        List of sample info, containing patient_id, split, mri_path, seg_path.

    Notes
    -----
    - No augmentation, no training-related operations.
    - Spatial preprocessing + intensity preprocessing only.
    - CUDA acceleration used when available.
    """

    def __init__(
        self,
        split: str = 'train',
        target_spacing: float = 1.0
    ):
        super().__init__()
        self.split = split
        self.target_spacing = target_spacing

        # Scan all patient folders
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
        Load and preprocess a single patient sample.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, Dict]
            volume : torch.Tensor, shape = (4, D, H, W), float32
            seg    : torch.Tensor, shape = (D, H, W), long int (4-class labels)
            info   : Dict, containing patient_id, split, shape etc.
        """
        sample = self.samples[idx]

        # ---------- Preprocess 4 MRI sequences ----------
        t1_data    = preprocess_single_mri(sample['t1'],    self.target_spacing)
        t1ce_data  = preprocess_single_mri(sample['t1ce'],  self.target_spacing)
        t2_data    = preprocess_single_mri(sample['t2'],    self.target_spacing)
        flair_data = preprocess_single_mri(sample['flair'], self.target_spacing)

        # ---------- Preprocess segmentation label ----------
        seg_data = preprocess_seg_label(sample['seg'], self.target_spacing)

        # ---------- Stack to 4-channel MRI ----------
        volume = build_4channel_input(t1_data, t1ce_data, t2_data, flair_data)

        # ---------- Convert to PyTorch Tensor ----------
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


# ============================================================================
# 7. Dataset Information Printing Functions
# ============================================================================

def print_dataset_info(split: str = 'all'):
    """
    Print dataset information and first 3 samples for each split.

    Parameters
    ----------
    split : str
        Which split to print: 'train', 'val', 'test', or 'all' (default).

    Returns
    -------
    None
        Prints information to console.
    """
    label_names = {
        0: 'Background (0)',
        1: 'NCR (1)',         # Necrotic tumor core
        2: 'ED (2)',          # Peritumoral edema
        3: 'ET (3)'           # GD-enhancing tumor
    }

    splits_to_process = ['train', 'val', 'test'] if split == 'all' else [split]

    print("\n" + "=" * 70)
    print("  BrainMRI Dataset Information - Data Preparation Module")
    print("=" * 70)
    print(f"  Device: {DEVICE.type.upper()}")
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # ===== Part 1: Dataset Overview =====
    print("\n" + "-" * 70)
    print("  Dataset Overview")
    print("-" * 70)

    for split_name in splits_to_process:
        try:
            dataset = BrainMRIDataset(split=split_name, target_spacing=1.0)
            print(f"\n  [{split_name.upper()}]")
            print(f"    Number of patients  : {len(dataset)}")
            print(f"    Target spacing      : {dataset.target_spacing} mm (isotropic)")

            # Count total labels across all samples
            all_labels = {0: 0, 1: 0, 2: 0, 3: 0}
            all_shapes = []

            for i in range(len(dataset)):
                _, seg, info = dataset[i]
                all_shapes.append(info['shape'])
                
                # Count labels using CUDA if available
                if torch.cuda.is_available():
                    seg_gpu = seg.to(DEVICE)
                    for label in [0, 1, 2, 3]:
                        all_labels[label] += (seg_gpu == label).sum().item()
                else:
                    for label in [0, 1, 2, 3]:
                        all_labels[label] += (seg == label).sum().item()

            # Print label distribution
            total_voxels = sum(all_labels.values())
            print(f"    Label distribution:")
            for label, count in all_labels.items():
                pct = count / total_voxels * 100 if total_voxels > 0 else 0
                print(f"      {label_names[label]:20s}: {count:>12,} voxels ({pct:6.2f}%)")

            # Print shape range
            all_shapes = np.array([list(s) for s in all_shapes])
            print(f"    Volume shape range:")
            print(f"      Channel : {int(all_shapes[:, 0].min())} (fixed)")
            print(f"      Depth   : {int(all_shapes[:, 1].min())} - {int(all_shapes[:, 1].max())}")
            print(f"      Height  : {int(all_shapes[:, 2].min())} - {int(all_shapes[:, 2].max())}")
            print(f"      Width   : {int(all_shapes[:, 3].min())} - {int(all_shapes[:, 3].max())}")

        except Exception as e:
            print(f"\n  [{split_name.upper()}]: Error - {e}")

    # ===== Part 2: First 3 Samples Detail =====
    print("\n" + "-" * 70)
    print("  First 3 Samples Detail (per split)")
    print("-" * 70)

    for split_name in splits_to_process:
        try:
            dataset = BrainMRIDataset(split=split_name, target_spacing=1.0)
            n_show = min(3, len(dataset))

            print(f"\n  [{split_name.upper()}] - Showing {n_show} sample(s):")

            for i in range(n_show):
                vol, seg, info = dataset[i]
                labels = info['unique_labels']
                label_detail = ', '.join(label_names.get(l, f'Unknown ({l})') for l in labels)

                print(f"\n    Patient {info['patient_id']}:")
                print(f"      Volume shape : {tuple(vol.shape)}  (C, D, H, W)")
                print(f"      Seg shape    : {tuple(seg.shape)}  (D, H, W)")
                print(f"      Unique labels: {labels}")
                print(f"      Label names  : {label_detail}")

                # Print data dtype and device info
                print(f"      Volume dtype : {vol.dtype}")
                print(f"      Seg dtype    : {seg.dtype}")

                # Print memory usage
                vol_mem = vol.element_size() * vol.numel() / 1024**2  # MB
                seg_mem = seg.element_size() * seg.numel() / 1024**2  # MB
                print(f"      Volume memory: {vol_mem:.2f} MB")
                print(f"      Seg memory   : {seg_mem:.2f} MB")

        except Exception as e:
            print(f"\n  [{split_name.upper()}]: Error - {e}")

    print("\n" + "=" * 70)
    print("  Data preparation complete. Ready for training pipeline.")
    print("=" * 70 + "\n")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Print complete dataset information
    print_dataset_info(split='all')
