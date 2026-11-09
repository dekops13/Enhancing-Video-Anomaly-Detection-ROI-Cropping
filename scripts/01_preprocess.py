"""RECONSTRUCTED preprocessing: YOLOv5 + OpenCV MOG2 gating -> letterboxed 32x32 ROIs.

The exact original detector weights, BSMOG2 variant and gating hyperparameters
are lost. This is an independent executable approximation, not the original code.
Use an installed YOLOv5 checkout and a compatible weights file, acquired separately.
"""
import argparse
import csv
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roi_vad.io import list_sequences, sequence_name, iter_frames, letterbox_grayscale


def build_detector(repository, weights, confidence, device):
    if not Path(repository).is_dir():
        raise FileNotFoundError(f'YOLOv5 checkout not found: {repository}')
    if not Path(weights).is_file():
        raise FileNotFoundError(f'Detector weights not found: {weights}')
    # Local checkout avoids implicit downloads from GitHub at runtime.
    model = torch.hub.load(str(repository), 'custom', path=str(Path(weights).resolve()),
                           source='local', trust_repo=True)
    model.conf = confidence
    model.to(device).eval()
    return model


def candidates(detector, frame, min_conf, classes):
    # Model's predicted boxes are [x1,y1,x2,y2,confidence,COCO_class].
    outputs = detector(frame).xyxy[0].detach().cpu().numpy()
    for x1, y1, x2, y2, conf, cls in outputs:
        if conf < min_conf or (classes and int(cls) not in classes):
            continue
        yield int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='Raw train or test sequences: images or videos')
    p.add_argument('--output', required=True, help='Processed root, e.g. data/processed/ped1/train')
    p.add_argument('--yolov5-repo', required=True, help='Local ultralytics/yolov5 checkout')
    p.add_argument('--weights', required=True, help='YOLOv5 .pt weights; NOT AE checkpoint')
    p.add_argument('--device', default='cpu')
    p.add_argument('--confidence', type=float, default=0.25)
    p.add_argument('--class-ids', default='0', help='COCO IDs comma-separated; 0=person, empty=all')
    p.add_argument('--mog-history', type=int, default=500)
    p.add_argument('--mog-var-threshold', type=float, default=16.)
    p.add_argument('--fg-ratio', type=float, default=0.02,
                   help='Minimum foreground occupancy inside a box')
    p.add_argument('--warmup', type=int, default=0, help='Frames to skip before saving ROIs')
    p.add_argument('--mask-foreground', action='store_true',
                   help='Apply foreground mask to crop; optional unverified experiment choice')
    args = p.parse_args()
    if not 0 <= args.fg_ratio <= 1:
        p.error('--fg-ratio must be in [0,1]')
    classes = {int(x) for x in args.class_ids.split(',') if x.strip()}
    output = Path(args.output)
    rois_dir = output / 'rois'
    rois_dir.mkdir(parents=True, exist_ok=True)
    detector = build_detector(args.yolov5_repo, args.weights, args.confidence, args.device)
    roi_rows = []
    frame_rows = []
    for sequence in list_sequences(args.input):
        name = sequence_name(sequence)
        bg = cv2.createBackgroundSubtractorMOG2(history=args.mog_history,
                         varThreshold=args.mog_var_threshold, detectShadows=False)
        (rois_dir/name).mkdir(parents=True, exist_ok=True)
        frame_count = crop_count = 0
        for frame_index, frame, source_path in iter_frames(sequence):
            frame_count += 1
            mask = bg.apply(frame)
            saved = 0
            if frame_index > args.warmup:
                h, w = frame.shape[:2]
                for box_index, (x1, y1, x2, y2, conf, cls) in enumerate(
                        candidates(detector, frame, args.confidence, classes)):
                    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
                    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    local_mask = mask[y1:y2, x1:x2]
                    ratio = float(np.mean(local_mask > 0))
                    if ratio < args.fg_ratio:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    if args.mask_foreground:
                        crop = cv2.bitwise_and(crop, crop, mask=local_mask)
                    image = letterbox_grayscale(crop, 32)
                    filename = f'{name}_f{frame_index:06d}_r{box_index:03d}.png'
                    roi_path = rois_dir/name/filename
                    if not cv2.imwrite(str(roi_path), image):
                        raise IOError(f'Failed to write ROI image: {roi_path}')
                    roi_rows.append(dict(sequence=name,frame_index=frame_index,
                        roi_path=str(roi_path.resolve()),source_path=source_path,
                        x1=x1,y1=y1,x2=x2,y2=y2,confidence=conf,class_id=cls,
                        fg_ratio=ratio))
                    saved += 1
                    crop_count += 1
            frame_rows.append(dict(sequence=name,frame_index=frame_index,
                source_path=source_path,num_rois=saved))
        print(f'{name}: {frame_count} source frames, {crop_count} saved ROIs')
    for filename, rows, fields in (
        ('roi_manifest.csv', roi_rows, ['sequence','frame_index','roi_path','source_path',
                   'x1','y1','x2','y2','confidence','class_id','fg_ratio']),
        ('frame_manifest.csv', frame_rows, ['sequence','frame_index','source_path','num_rois'])):
        with (output/filename).open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh,fieldnames=fields)
            writer.writeheader();writer.writerows(rows)
    print(f'Saved {len(roi_rows)} ROI records and {len(frame_rows)} frame records to {output}')

if __name__ == '__main__':
    main()
