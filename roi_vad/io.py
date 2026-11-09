"""Dataset frame streams and 1-based frame indices (new reconstruction)."""
from pathlib import Path
import re
import cv2

IMAGE_EXTS = {'.bmp', '.png', '.tif', '.tiff', '.jpg', '.jpeg'}
VIDEO_EXTS = {'.avi', '.mp4', '.mov', '.mkv'}


def natural_key(path):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r'(\d+)', Path(path).name)]


def list_sequences(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'Input directory does not exist: {root}')
    items = [p for p in root.iterdir() if p.is_dir() or p.suffix.lower() in VIDEO_EXTS]
    if items:
        return sorted(items, key=natural_key)
    if any(p.suffix.lower() in IMAGE_EXTS for p in root.iterdir()):
        return [root]
    raise ValueError(f'No frame-sequence folders or video files found under: {root}')


def sequence_name(path):
    return Path(path).stem


def iter_frames(sequence):
    """Yield (1-based sequential frame index, BGR ndarray, optional source frame path).

    Index is derived from temporal order, NEVER by the first digits in crop names.
    """
    sequence = Path(sequence)
    if sequence.is_dir():
        frames = sorted((p for p in sequence.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=natural_key)
        for idx, p in enumerate(frames, start=1):
            frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f'Unreadable frame: {p}')
            yield idx, frame, str(p.resolve())
    elif sequence.suffix.lower() in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(sequence))
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {sequence}')
        i = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                i += 1
                yield i, frame, ''
        finally:
            cap.release()
    else:
        raise ValueError(f'Unsupported sequence: {sequence}')


def letterbox_grayscale(roi, size=32):
    """Center letterbox for aspect preservation; exact original padding unknown."""
    if roi is None or roi.size == 0:
        raise ValueError('Empty ROI')
    h, w = roi.shape[:2]
    s = min(size/w, size/h)
    nw, nh = max(1, min(size, round(w*s))), max(1, min(size, round(h*s)))
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    import numpy as np
    canvas = np.zeros((size, size), dtype=np.uint8)
    left, top = (size-nw)//2, (size-nh)//2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas
