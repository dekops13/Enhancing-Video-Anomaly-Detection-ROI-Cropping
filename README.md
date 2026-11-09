# Enhancing Video Anomaly Detection through Optimized Pre-processing

**ROI Cropping and Reconstruction Loss** — Python implementation of the methodology reported by Saefudin, Jasmir, and Putra at ICAISD 2025.

**Publication:** [DOI: 10.1109/ICAISD68166.2025.11385780](https://doi.org/10.1109/ICAISD68166.2025.11385780)

This repository provides a practical, documented implementation of a YOLOv5–MOG2–convolutional-autoencoder pipeline for **UCSD Ped1, UCSD Ped2, and CUHK Avenue**. It supports CPU execution, Jupyter notebooks, validation-based threshold calibration, and frame-level evaluation.

> **Reproducibility:** The code follows the published methodology. It is not an archival copy of the original experimental scripts or checkpoint weights, which are unavailable. Several parameters were not specified in the article. The reference results have not yet been independently reproduced; see [experimental protocol](docs/EXPERIMENTAL_PROTOCOL.md).

## How it works

`Video frames → YOLOv5 detections → MOG2 foreground gate → 32×32 ROI patches → CNN autoencoder → reconstruction MSE → validation-calibrated threshold → frame-level predictions`

The autoencoder is trained **only on normal ROI patches**. During evaluation, pixelwise mean squared error measures the difference between a patch and its reconstruction. Multiple ROI scores from one frame are combined using the maximum (a documented implementation choice).

## Project structure

```text
configs/                   Dataset reference configurations
roi_vad/                   Model, frame input, ground truth, evaluation helpers
scripts/
  01_preprocess.py         Object detection, foreground gate, ROI crops
  02_train.py              Normal-only autoencoder training
  03_evaluate.py           Threshold calibration and test evaluation
  04_compare_paper.py      Comparison against published reference values
  05_qualitative.py        TP / FP / FN examples when original frames exist
notebooks/
  video_anomaly_detection.ipynb
  docs/                    Experimental protocol and caveats
requirements.txt
README.md
```

## 1. Installation

Python 3.10 or 3.11 is recommended. Run from the repository root:

```bash
git clone https://github.com/dekops13/Enhancing-Video-Anomaly-Detection-ROI-Cropping.git
cd Enhancing-Video-Anomaly-Detection-ROI-Cropping
conda create -n roi-vad python=3.10 -y
conda activate roi-vad
python -m pip install -r requirements.txt
python -m pip install jupyterlab
python -m pytest -q
```

Install a **separate local** copy of [YOLOv5](https://github.com/ultralytics/yolov5), including its requirements, and download compatible pretrained detection weights (e.g., `yolov5s.pt`). These are not the autoencoder weights and are not distributed here. The exact detector release/weights in the original study are unspecified.

## 2. Prepare benchmark datasets

Download datasets from their respective providers. Organize original frames by sequence as below (examples for Windows):

```text
C:/datasets/
  UCSDped1/
    Train/Train001/001.tif ...
    Test/Test001/001.tif ...
    Test_gt/Test003_gt/001.tif ...
  UCSDped2/
    Train/Train001/001.tif ...
    Test/Test001/001.tif ...
    Test_gt/Test001_gt/001.tif ...
  Avenue/
    Train/01/0001.png ...
    Test/01/0001.png ...
    testing_label_mask/1_label.mat ...
```

Frame folders and video files (`.avi`, `.mp4`, etc.) are supported as preprocessing inputs. The loader requires correct sequence names and chronological frame order. Ground-truth masks must match the exact number of source frames; sequences without annotations are excluded. Dataset layouts can differ by distribution. When generating qualitative figures from video files, first extract frames as images so source image paths are available.

## 3. Run a full experiment

The following commands are for **UCSD Ped1**. Use forward slashes in Windows paths or replace paths with your own. The working directory must be the repository root.

### Extract motion-relevant ROIs

```bash
python scripts/01_preprocess.py --input "C:/datasets/UCSDped1/Train" --output "data/Ped1/train" --yolov5-repo "C:/code/yolov5" --weights "C:/weights/yolov5s.pt" --device cpu
python scripts/01_preprocess.py --input "C:/datasets/UCSDped1/Test" --output "data/Ped1/test" --yolov5-repo "C:/code/yolov5" --weights "C:/weights/yolov5s.pt" --device cpu
```

Both commands save `roi_manifest.csv`, `frame_manifest.csv`, and `rois/<sequence>/*.png`. The motion gate uses configurable default values (`--confidence 0.25`, `--mog-history 500`, `--mog-var-threshold 16`, `--fg-ratio 0.02`); these values are **examples**, not historically verified settings.

### Train the convolutional autoencoder

```bash
python scripts/02_train.py --rois "data/Ped1/train/rois" --output "checkpoints/Ped1/best_model.pth" --epochs 50 --batch-size 128 --device cpu
```

The training code reserves whole normal **training sequences** for reconstruction-loss validation, uses Adam/MSE and early stopping, and writes the best checkpoint and its training log. The learning rate and patience are configurable examples; inspect `python scripts/02_train.py --help` to adjust them.

### Evaluate using validation-only threshold calibration

```bash
python scripts/03_evaluate.py --dataset Ped1 --processed-test "data/Ped1/test" --gt-root "C:/datasets/UCSDped1/Test_gt" --checkpoint "checkpoints/Ped1/best_model.pth" --result-dir "results/Ped1" --threshold-mode p95 --device cpu
```

`--threshold-mode p95` uses the 95th percentile of **validation** reconstruction scores. Other choices are `meanstd --k 3` and `f1`; the latter requires both classes in the validation set. The split is performed by sequence, without using test scores for threshold tuning. The alternative rules and publication discrepancies are described in [experimental protocol](docs/EXPERIMENTAL_PROTOCOL.md).

## 4. Other datasets

For **UCSD Ped2**, repeat the three stages using `C:/datasets/UCSDped2/Train`, `Test`, and `Test_gt`, output directories under `data/Ped2/`, `checkpoints/Ped2/`, `results/Ped2/`, and `--dataset Ped2`.

For **CUHK Avenue**, use `C:/datasets/Avenue/Train`, `Test`, and `testing_label_mask`, output directories under `data/Avenue/`, `checkpoints/Avenue/`, `results/Avenue/`, and `--dataset Avenue`. Its MATLAB masks are supported, but GT length and sequence IDs must be verified. Some distributions lack matching labels for the last test sequence; the evaluator excludes those cases rather than inventing labels.

## 5. Results and visualizations

The evaluator writes:

```text
results/Ped1/
  evaluation_summary.json       Threshold, metrics, sequence split, coverage
  validation_frames.csv         Validation frame scores and labels
  test_frames.csv               Test frame IDs, scores, labels, predictions
  scores.npy                    Test frame-level reconstruction scores
  labels.npy                    Aligned test frame-level ground-truth labels
  roc_curve.png                 ROC plot
  pr_curve.png                  Precision–recall plot
  validation_threshold_sweep.csv  F1 sweep, if F1 threshold is selected
```

Compare benchmark measurements with the numbers reported in the article:

```bash
python scripts/04_compare_paper.py --results results
python scripts/05_qualitative.py --result-dir results/Ped1
```

The qualitative script requires valid source frame image paths and at least one TP, FP and FN. It fails explicitly if those cases are unavailable.

## 6. Jupyter workflow

Start JupyterLab from the repository root and open `notebooks/video_anomaly_detection.ipynb`. Update dataset and YOLOv5 paths in the configuration cell and run the cells in order.

```bash
jupyter lab
```

## Known limitations

- The pretrained YOLOv5 detector and original autoencoder checkpoints are not distributed.
- Exact training split IDs, detection parameters, augmentation and some hyperparameters are unspecified in the article.
- Maximum ROI pooling and handling of frames with no detected ROI are explicit implementation choices; frame coverage is reported.
- Unannotated sequences are not classified as normal. If GT/frame alignment fails, evaluation stops instead of silently assigning labels.
- The published performance values are reference results, not automatically recreated by these scripts.

See [experimental protocol](docs/EXPERIMENTAL_PROTOCOL.md) for benchmark reference values and a complete list of required choices. If you use this code in a publication, cite the [ICAISD paper](https://doi.org/10.1109/ICAISD68166.2025.11385780) and report detector parameters, dataset version, training split, threshold rule and evaluation coverage.
