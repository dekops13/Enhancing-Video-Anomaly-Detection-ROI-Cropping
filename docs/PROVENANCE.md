# Evidence ledger — original research source recovery

Status: PARTIAL RECOVERY, NOT A REPRODUCIBLE RELEASE.

## Verified as user-executed from conversation logs
- Ped1 checkpoint `models\\best_modelped1aug.pth` loaded successfully.
- Ped1 matched-GT run: total ROI sequence directories=36, matched GT sequences=10, VAL=3, TEST=7; VAL samples=2636/anomalies=839; threshold=0.002526, validation F1=0.3217; TEST samples=3889/anomalies=1733, P=.4657, R=.3687, F1=.4116, AUC=.4952, CM=[[1423,733],[1094,639]]. `scores.npy` and `labels.npy` saved.
- Earlier Ped1 run VAL samples=1797/anomalies=0, TEST samples=4728/anomalies=2572, threshold fallback=.006349, AUC=.5079. This is a DIFFERENT, problematic protocol and is NOT the final matched-GT run.
- Avenue executed earlier with 20879 ROI samples, 3544 anomaly labels, threshold=.010444, P=.2155, R=.0635, F1=.0981, AUC=.5391. Its GT/ROI alignment was challenged in conversation; do not assert frame-level validity without verification.

## Source present in chat
- `CNNAutoencoder` class code (extracted verbatim into source_from_chat/cnn_autoencoder.py).
- UCSD `parse_frame_index` and GT loader (extracted verbatim into source_from_chat/ucsd_gt_loader.py).
- Multiple full validation/evaluation code versions were pasted; the exact complete code that generated the matched-GT Ped1 output was NOT pasted as one verifiable final script in the currently accessible record.

## Source NOT recovered; must not fabricate
- YOLOv5 extraction and BSMOG2 foreground-gating implementation.
- Complete training/augmentation loops, split manifests, seeds, optimizer hyperparameters as executed, and original Jupyter notebook.
- `best_modelped1aug.pth`, Ped2/Avenue checkpoints, raw `scores.npy` / `labels.npy`, raw datasets.
- Exact executed Ped2 and Avenue final scripts with GT/frame alignment verification.

## Methodological discrepancies to resolve
- Published paper describes up to 50 epochs, batch 128, Adam/MSE, 95th-percentile validation threshold.
- Some executed chat evaluation scripts select F1-optimal threshold on labeled validation sequences.
- Multiple ROIs per video frame must be aggregated before claiming frame-level ROC/AUC. Saved arrays described as per-ROI cannot automatically be called frame-level metrics.
- Dataset statistics in the paper may be ROI sample counts, not unique video-frame counts; verify against manifests.
- Published paper table reports Ped1 P=.6998/R=.3761/F1=.4893/AUC=.5324; Ped2 P=.9305/R=.4934/F1=.6449/AUC=.5476; Avenue P=.1956/R=.1428/F1=.1651/AUC=.5391. These are NOT identical to earlier executed logs and must not be represented as reproduced by this archive.

## Next recovery locations
- Local Windows `C:\\Users\\dekops\\miniconda3\\` and associated Jupyter notebook directories.
- `models\\best_modelped1aug.pth`, `models\\best_modelped2aug.pth`, `Avenruns\\roi_ae32\\avenue_best_model.pth` (candidate names, verify on disk).
- `Dataset\\UCSDped1\\Yolo5LocalOutput\\crops`, equivalent Ped2 and Avenue folders.
- Original `main.ipynb` referenced in past chats but not retrieved from Drive.
