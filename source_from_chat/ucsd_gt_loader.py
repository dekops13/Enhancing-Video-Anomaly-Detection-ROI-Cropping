# SOURCE PROVENANCE: functions copied from user's UCSD Ped1 validation script.
# NOTE: original loader; no new fixes introduced. Test_gt contains only 10 sequence folders for Ped1.
import re
import cv2
from pathlib import Path

UCSD_ROOT = r"C:\Users\dekops\miniconda3\Dataset\UCSDped1"

def parse_frame_index(filename: str):
    # cocokkan pola "_f00012" → 12
    m = re.search(r"_f(\d+)", filename)
    return int(m.group(1)) if m else None

def gt_labels_for_seq(seq_name: str):
    """Buat label per-frame untuk satu sequence berdasarkan mask Test_gt\\<seq>_gt\\*."""
    gt_dir = Path(UCSD_ROOT) / "Test_gt" / f"{seq_name}_gt"
    if not gt_dir.exists(): return []
    masks = sorted([p for p in gt_dir.iterdir() if p.is_file()])
    labels = []
    for m in masks:
        im = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        labels.append(1 if (im is not None and cv2.countNonZero(im) > 0) else 0)
    return labels
