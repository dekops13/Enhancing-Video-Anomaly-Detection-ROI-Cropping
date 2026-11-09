# Experimental protocol and reproducibility notes

## Scope

This repository implements the video anomaly detection method described in Saefudin, Jasmir, and Putra, *Enhancing Video Anomaly Detection through Optimized Pre-processing: ROI Cropping and Reconstruction Loss*, ICAISD 2025, DOI: [10.1109/ICAISD68166.2025.11385780](https://doi.org/10.1109/ICAISD68166.2025.11385780).

It is a **reference implementation developed from the published method**, not an archival copy of the original experimental environment. Trained checkpoints, split manifests, original preprocessing settings, and detector weights are not included. The published metric values have not been independently reproduced with this repository. Researchers should report their newly measured results and parameters.

## Methods specified by the publication

- Input: 32 x 32 grayscale ROI patches, resized and normalized.
- Object detection: YOLOv5; motion gating: MOG2 background subtraction.
- Encoder: 3 convolutional layers of 32, 64, 128 channels, stride 2, kernel size 3.
- Bottleneck: 128-dimensional dense representation.
- Decoder: 3 transposed convolutions, sigmoid grayscale output.
- Training: normal-only data, Adam optimizer, pixelwise MSE, up to 50 epochs, batch size 128, early stopping.
- Benchmarks: UCSD Ped1, UCSD Ped2, CUHK Avenue.

## Parameters requiring explicit reporting

The publication does not determine the precise YOLOv5 commit and weights, detection confidence, MOG2 history/variance threshold, motion overlap cutoff, training learning rate/patience, augmentation, seed, original validation/test sequence IDs, per-frame pooling function, or no-ROI policy. The CLI exposes example settings; these must **not** be treated as recovered experimental parameters.

## Threshold strategies

The paper contains three different threshold descriptions: validation P95 (Experimental Setup), validation mean plus k standard deviations (abstract and diagram), and F1-maximizing labeled validation selection (Methods). Scripts support all three through explicit `--threshold-mode`. Scores from the test split are never used for calibration.

## Evaluation unit and ground truth

This implementation aggregates the largest MSE among ROIs associated with each original frame. It checks frame/GT lengths, excludes sequences without GT, and records coverage when frames have no detected ROI. The maximum pooling and frame skipping defaults are implementation choices, not confirmed original settings. Scores, labels, and frame-index manifests are written together to prevent ambiguous alignment. For Avenue, test sequences lacking a matching ground-truth file are excluded rather than labeled normal.

## Results reported in the article (reference values only)

| Dataset | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|
| UCSD Ped1 | 0.6998 | 0.3761 | 0.4893 | 0.5324 |
| UCSD Ped2 | 0.9305 | 0.4934 | 0.6449 | 0.5476 |
| CUHK Avenue | 0.1956 | 0.1428 | 0.1651 | 0.5391 |

Use `scripts/04_compare_paper.py` to compare newly measured metrics to these reference values. A numerical match should not be claimed without verifying all underlying evaluation choices and benchmark coverage.
