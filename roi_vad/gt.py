"""Ground-truth loaders for labeled UCSD Ped1/Ped2 and CUHK Avenue sequences.

No masks or undecodable labels are fabricated as negatives. Missing annotation
must be excluded or cause an explicit error. Avenue MAT structures vary.
"""
from pathlib import Path
import numpy as np
import cv2
from scipy.io import loadmat
from .io import natural_key, IMAGE_EXTS


def _sorted_masks(folder):
    return sorted((p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=natural_key)


def _image_labels(folder):
    paths = _sorted_masks(folder)
    if not paths:
        raise ValueError(f'No image masks in {folder}')
    labels = []
    for p in paths:
        image = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f'Unreadable GT mask: {p}')
        labels.append(int(np.any(image > 0)))
    return np.asarray(labels, dtype=np.int8)


def _mat_mask_frames(path):
    """Extract MATLAB mask cell array or 3D mask stack, no spatial flattening.

    A 2D numeric matrix is a SINGLE SPATIAL MASK, not a length H*W label vector.
    """
    data = loadmat(str(path))
    candidates = [(k, v) for k, v in data.items() if not k.startswith('__')]
    candidates.sort(key=lambda kv: 0 if kv[0].lower() in ('vollabel', 'label', 'labels', 'gt') else 1)
    for name, val in candidates:
        if not isinstance(val, np.ndarray):
            continue
        if val.dtype == object:
            cells = [np.asarray(v).squeeze() for v in val.ravel(order='F')]
            if cells and all(v.ndim == 2 and v.size >= 64 and np.issubdtype(v.dtype, np.number) for v in cells):
                return np.asarray([int(np.any(v > 0)) for v in cells], dtype=np.int8)
        elif val.ndim == 3 and np.issubdtype(val.dtype, np.number):
            # Avenue video is HxWxT in common MATLAB distribution.
            if val.shape[0] > 64 and val.shape[1] > 64:
                return np.asarray([int(np.any(val[:, :, i] > 0)) for i in range(val.shape[2])], dtype=np.int8)
            if val.shape[1] > 64 and val.shape[2] > 64:
                return np.asarray([int(np.any(val[i] > 0)) for i in range(val.shape[0])], dtype=np.int8)
    raise ValueError(f'Cannot decode frame mask series from {path}; inspect MATLAB keys and shapes')


def load_labels(dataset, gt_root, sequence):
    """Return 1D frame labels (GT temporal order, 1-based externally), or None if no GT.

    `sequence` is 'Test003' for UCSD, '01' for Avenue.
    """
    gt_root = Path(gt_root)
    if dataset.lower() in ('ped1', 'ped2'):
        folder = gt_root / f'{sequence}_gt'
        return _image_labels(folder) if folder.is_dir() else None
    if dataset.lower() != 'avenue':
        raise ValueError(f'Unknown dataset: {dataset}')
    for name in (sequence, str(int(sequence)).zfill(2) if sequence.isdigit() else sequence):
        folder = gt_root / name
        if folder.is_dir():
            return _image_labels(folder)
    if sequence.isdigit():
        seq_number = int(sequence)
        for name in (f'{seq_number}_label.mat', f'{seq_number:02d}_label.mat'):
            path = gt_root / name
            if path.is_file():
                return _mat_mask_frames(path)
    return None
