# Enhancing Video Anomaly Detection through Optimized Pre-processing: ROI Cropping and Reconstruction Loss

[![Status: reconstructed research implementation](https://img.shields.io/badge/status-reconstructed%20implementation-orange)](docs/PROVENANCE.md)

**Publication:** D. F. Saefudin, Jasmir, and Z. P. Putra, “Enhancing Video Anomaly Detection through Optimized Pre-processing: ROI Cropping and Reconstruction Loss,” *2025 International Conference on Advanced Information Scientific Development (ICAISD)*. **DOI:** [10.1109/ICAISD68166.2025.11385780](https://doi.org/10.1109/ICAISD68166.2025.11385780).

## What this repository is — and is not

This project provides a **reconstructed, auditable Python implementation** of the paper's general workflow for the UCSD Ped1, UCSD Ped2, and CUHK Avenue datasets. It includes a few **original snippets copied from the authors' past research conversations**, stored separately under `original_chat_excerpts/` and `source_from_chat/`.

**This is not the lost original experiment source code, and it has not reproduced the paper's published metric values.** The original preprocessing/training notebooks, trained autoencoder checkpoints, precise YOLOv5 detector version/weights, and some experimental settings have not been recovered. Those gaps are tracked in [PROVENANCE.md](docs/PROVENANCE.md) and [PUBLICATION_AUDIT.md](docs/PUBLICATION_AUDIT.md). Do not cite generated results as a reproduction until the complete protocol has been verified.

## Pipeline

1. **Preprocessing:** run a separately installed local YOLOv5 detector on frames; run OpenCV MOG2 foreground subtraction; retain motion-relevant detections; crop and letterbox each ROI to a **32 × 32 grayscale image**.
2. **Training:** train a convolutional autoencoder on **normal-only** ROI patches using reconstruction **mean squared error (MSE)**. The encoder uses three stride-2 convolution layers with 32, 64, and 128 channels and a 128-dimensional bottleneck; the decoder mirrors the spatial upsampling.
3. **Validation:** score annotated validation video sequences and select an anomaly threshold using one explicitly chosen rule (`p95`, `meanstd`, or labeled-validation `f1`). **No test-set threshold tuning.**
4. **Testing:** calculate ROI reconstruction MSE, pool by `max` to one score per original frame, compare the score with the frozen threshold, and compute Precision, Recall, F1, ROC-AUC, confusion matrix, and true ROC/PR plots.
5. **Audit:** save sequence-aware manifests and compare new results against the published table without claiming equivalence.

## Repository layout

```text
configs/                 Example configurations and unknown parameters
roi_vad/                 Reconstructed model, I/O, ground-truth, evaluation code
scripts/01_preprocess.py YOLOv5 + MOG2 → ROI + frame/ROI manifests
scripts/02_train.py      Normal-only CNN autoencoder training
scripts/03_evaluate.py   Validation threshold + frame-level evaluation
scripts/04_compare_paper.py Publication table comparison
scripts/05_qualitative.py Genuine TP/FP/FN frame collage (when available)
notebooks/ROI_VAD_Reconstructed_Experiment.ipynb
original_chat_excerpts/  Author-pasted earlier code — preserved separately
tests/test_pipeline.py  Synthetic unit tests
```

## System requirements

- **Python 3.10 or 3.11** recommended, Windows/Linux/macOS; sufficient disk space for video datasets and generated crops.
- PyTorch + OpenCV; **CPU inference and training supported** but may be slow. NVIDIA CUDA is optional when installed correctly.
- **A separate YOLOv5 checkout and compatible pretrained detection `.pt` weights** are required. They are **not** included and are **not** the same as the autoencoder `.pth` files.
- Public benchmark datasets must be obtained independently, respecting their own licenses and terms.
- Use the repository root as the working directory for commands below.

### Installation (Windows Miniconda / Anaconda)

Open **Anaconda Prompt** (or PowerShell if conda is configured):

```powershell
conda create -n roi-vad python=3.10 -y
conda activate roi-vad
python -m pip install -r requirements.txt
python -m pip install jupyterlab
```

**YOLOv5 setup:** download or clone a compatible YOLOv5 source checkout into a separate folder (for example `C:/code/yolov5`). Install the dependencies specified by that checkout. Obtain a detector weight file such as `yolov5s.pt` from the official project as appropriate, and point the command-line arguments to both local paths. The exact original detector checkpoint is unknown. See <https://github.com/ultralytics/yolov5> for the upstream software and its license.

A quick integrity check (synthetic tests, **not a full reproducibility run**):

```powershell
python -m pytest -q
```

## Organize your datasets

This example uses the **original unprocessed frames**. The system does not infer labels from ROI file order.

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
    Train/01/*.png ...      # or decoded video frames arranged by sequence
    Test/01/*.png ...
    testing_label_mask/1_label.mat ...  # or 01/0001.png ...
```

The folder tree above is **illustrative**, not a guarantee about the file layout in your particular download. The supplied code reads image-sequence folders or video files for preprocessing; for qualitative plotting, decoded original frame files are necessary. UCSD GT often exists for only some sequences: missing GT is excluded, **never treated as normal**. Avenue `.mat` masks must be checked for matching temporal length and sequence IDs; some distributions have different labels/sequence counts.

### Example 1 — UCSD Ped1 (PowerShell)

From the repository root, replace the paths for your own machine:

```powershell
python scripts/01_preprocess.py --input "C:/datasets/UCSDped1/Train" --output "data/Ped1/train" --yolov5-repo "C:/code/yolov5" --weights "C:/weights/yolov5s.pt" --device cpu
python scripts/01_preprocess.py --input "C:/datasets/UCSDped1/Test"  --output "data/Ped1/test"  --yolov5-repo "C:/code/yolov5" --weights "C:/weights/yolov5s.pt" --device cpu
python scripts/02_train.py --rois "data/Ped1/train/rois" --output "checkpoints/Ped1/reconstructed_best.pth" --epochs 50 --batch-size 128 --device cpu
python scripts/03_evaluate.py --dataset Ped1 --processed-test "data/Ped1/test" --gt-root "C:/datasets/UCSDped1/Test_gt" --checkpoint "checkpoints/Ped1/reconstructed_best.pth" --result-dir "results/Ped1" --threshold-mode f1 --device cpu
```

**Important:** `f1` above uses labeled validation sequences and matches **one** experiment variant discussed in the author's past conversations. It is **not interchangeable** with the publication's P95 or mean+standard-deviation descriptions. For a P95 sensitivity run, change only `--threshold-mode p95`. To test a mean-plus-three-standard-deviations rule, use `--threshold-mode meanstd --k 3`.

### Example 2 — UCSD Ped2

```powershell
python scripts/01_preprocess.py --input "C:/datasets/UCSDped2/Train" --output "data/Ped2/train" --yolov5-repo "C:/code/yolov5" --weights "C:/weights/yolov5s.pt" --device cpu
python scripts/01_preprocess.py --input "C:/datasets/UCSDped2/Test"  --output "data/Ped2/test"  --yolov5-repo "C:/code/yolov5" --weights "C:/weights/yolov5s.pt" --device cpu
python scripts/02_train.py --rois "data/Ped2/train/rois" --output "checkpoints/Ped2/reconstructed_best.pth" --epochs 50 --batch-size 128 --device cpu
python scripts/03_evaluate.py --dataset Ped2 --processed-test "data/Ped2/test" --gt-root "C:/datasets/UCSDped2/Test_gt" --checkpoint "checkpoints/Ped2/reconstructed_best.pth" --result-dir "results/Ped2" --threshold-mode f1 --device cpu
```

### Example 3 — CUHK Avenue

First provide your Avenue data in sequence folders, e.g. `C:/datasets/Avenue/Train/01/*.png` and `C:/datasets/Avenue/Test/01/*.png`; if your distribution consists of `.avi` files, use an appropriate input folder supported by the preprocessing script. Confirm that the GT `.mat` or PNG masks align with test sequence identifiers and frame counts.

```powershell
python scripts/01_preprocess.py --input "C:/datasets/Avenue/Train" --output "data/Avenue/train" --yolov5-repo "C:/code/yolov5" --weights "C:/weights/yolov5s.pt" --device cpu
python scripts/01_preprocess.py --input "C:/datasets/Avenue/Test"  --output "data/Avenue/test"  --yolov5-repo "C:/code/yolov5" --weights "C:/weights/yolov5s.pt" --device cpu
python scripts/02_train.py --rois "data/Avenue/train/rois" --output "checkpoints/Avenue/reconstructed_best.pth" --epochs 50 --batch-size 128 --device cpu
python scripts/03_evaluate.py --dataset Avenue --processed-test "data/Avenue/test" --gt-root "C:/datasets/Avenue/testing_label_mask" --checkpoint "checkpoints/Avenue/reconstructed_best.pth" --result-dir "results/Avenue" --threshold-mode p95 --device cpu
```

The above Avenue example uses P95 as a **protocol choice**; it does not imply the published table's Avenue score was obtained by this reconstructed code.

## Threshold selection and protocol consistency

**Select `--threshold-mode` explicitly**; the publication contains three descriptions that cannot all be treated as identical:

| Mode | How threshold is selected | Uses anomaly validation labels? |
|---|---|---|
| `p95` | 95th percentile of validation scores | No |
| `meanstd --k 3` | validation mean + 3 × validation standard deviation | No |
| `f1` | threshold maximizing labeled validation F1 | Yes |

Validation/test splits are at **sequence level**. AUC-ROC is threshold-independent; Precision, Recall, and F1 are threshold-dependent. The default reconstructed frame pooling is **maximum ROI MSE per frame**. Frames with no retained ROI are skipped by default and coverage is recorded; you can explicitly choose `--no-roi zero` for an alternative sensitivity study. Do not interpret these options as verified original methods.

## Outputs and review

The evaluation command produces `results/<Dataset>/` including:

```text
evaluation_summary.json       Metrics, split IDs, threshold, frame coverage
validation_frames.csv         One row per scored validation frame
test_frames.csv               Per-frame GT, score, prediction, case, frame identity
scores.npy                    Frame-level TEST scores, aligned to test_frames.csv
labels.npy                    Corresponding frame-level TEST 0/1 labels
roc_curve.png                 Real ROC curve (if both labels exist)
pr_curve.png                  Real precision–recall curve (if applicable)
validation_threshold_sweep.csv  Only for labeled F1 selection
```

When all three dataset runs finish:

```powershell
python scripts/04_compare_paper.py --results results
python scripts/05_qualitative.py --result-dir results/Ped1
```

The first command writes `results/comparison_with_paper.csv` containing differences from the published Table II. The second creates a **genuine TP/FP/FN collage** when all three categories exist **and** `test_frames.csv` contains valid original image paths. It will stop with a clear error rather than fabricate missing examples. Video-backed data may need conversion into frame-image sequences before qualitative plotting.

### Published reference metrics (NOT automatically reproduced)

| Dataset | Precision | Recall | F1 | AUC-ROC |
|---|---:|---:|---:|---:|
| UCSD Ped1 | 0.6998 | 0.3761 | 0.4893 | 0.5324 |
| UCSD Ped2 | 0.9305 | 0.4934 | 0.6449 | 0.5476 |
| CUHK Avenue | 0.1956 | 0.1428 | 0.1651 | 0.5391 |

These are **published figures**, not regenerated predictions. Reproducing them would require resolving the unavailable detector checkpoint, exact original split/masks, missing autoencoder weights, augmentation and training configuration, and differing threshold descriptions.

## Jupyter notebook

```powershell
jupyter lab
```

Open `notebooks/ROI_VAD_Reconstructed_Experiment.ipynb`. This is a **reconstructed workflow guide**, not the author's original missing `.ipynb` notebook. Read its configuration cells before executing any stage.

## Troubleshooting

- **`FileNotFoundError` for the ROI root:** inspect your actual folders; do not append `/Test` twice. The examples above use `.../Test` exactly once.
- **`VAL samples=0`:** confirm matched annotated sequence names, e.g., `Test003` ↔ `Test003_gt`, supported GT `.tif`/`.png`, and the ROI-to-frame manifest.
- **Empty ROC or AUC undefined:** no scored samples or only one GT class; inspect `evaluation_summary.json`, split IDs and coverage rather than forcing an arbitrary AUC.
- **PyTorch checkpoint shape mismatch:** the original `.pth` may belong to a different model; `strict=False` cannot fix mismatched tensor sizes. Check the saved architecture before loading.
- **YOLOv5 dependency failure:** install dependencies of the particular YOLOv5 checkout; the detector and CAE have separate checkpoints.
- **Metrics differ from the paper:** expected until the original split, weights and calibration protocol are recovered; compare them explicitly using `04_compare_paper.py`.

## Data, checkpoints and license

No raw benchmark video, GT masks, pretrained detector weights, original autoencoder checkpoints or result arrays are distributed here. Check dataset licenses and YOLOv5 licensing separately. Do not commit `data/`, `results/`, `checkpoints/`, credentials, or large model files. See `.gitignore`.

**Code license:** No license is granted by this documentation. A repository license should be selected explicitly by the repository owner before downstream reuse.

## Citation

```bibtex
@inproceedings{saefudin2025roi,
  title={Enhancing Video Anomaly Detection through Optimized Pre-processing: ROI Cropping and Reconstruction Loss},
  author={Saefudin, Dede Firmansyah and Jasmir and Putra, Zico Pratama},
  booktitle={2025 International Conference on Advanced Information Scientific Development (ICAISD)},
  year={2025},
  doi={10.1109/ICAISD68166.2025.11385780}
}
```

## Reproducibility and provenance

Please read [PROVENANCE](docs/PROVENANCE.md), [publication audit](docs/PUBLICATION_AUDIT.md), and [research recovery summary](docs/RECOVERY_SUMMARY.md) **before** describing the output as a replication of the published results.
