# Publication-to-code reproducibility audit

Primary source: **Enhancing Video Anomaly Detection through Optimized Pre-processing: ROI Cropping and Reconstruction Loss**, IEEE ICAISD 2025, DOI 10.1109/ICAISD68166.2025.11385780.

## Direct publication facts

- Input: grayscale, normalized, letterboxed ROI size 32x32.
- Detector: YOLOv5, background/foreground selection: BSMOG2 / OpenCV MOG2 (specific weights and thresholds **not provided**).
- CNN-AE encoder: Conv(1,32,3,2,1), Conv(32,64,3,2,1), Conv(64,128,3,2,1), FC 2048->128; decoder inverse FC and three ConvTranspose2d layers to 1x32x32; MSE reconstruction loss.
- Training: normal-only, up to 50 epochs, batch size 128, Adam, early stopping. Learning rate, augmentation recipe, exact patience, random seed **not reported**.
- Evaluation described as frame-level, but earlier log in conversation had per-ROI sample counts. Reconstructed scripts explicitly aggregate max ROI loss per frame; this pooling **is not verified as originally used**.
- Threshold description is inconsistent: abstract/figure/pipeline claim mean+k*std, method mentions labeled validation F1 optimization, Experimental Setup specifies validation P95. All three options are exposed. Do not combine or claim the original metric output for any of them.
- Published Table II: Ped1 P=.6998, R=.3761, F1=.4893, AUROC=.5324; Ped2 P=.9305, R=.4934, F1=.6449, AUROC=.5476; Avenue P=.1956, R=.1428, F1=.1651, AUROC=.5391.

## Provenance labels

- `original_chat_excerpts/`: copied from user-pasted conversation code; may represent historic/earlier versions, not necessarily the code producing published results.
- `source_from_chat/`: exact components from user messages.
- `roi_vad/` and `scripts/`: independently **RECONSTRUCTED** from the described pipeline; technically runnable, but not original source.
- `notebooks/`: executable wrapper around reconstructed CLI code, **not** original notebook.

## Known reproduction blockers

1. No original YOLOv5 weights or original foreground-gating configuration recovered.
2. No released original Conv-AE model checkpoint, trained weights, or normal training split manifest recovered.
3. Some GT folders exist only for annotated UCSD sequences; the evaluator excludes missing labels rather than making them normal.
4. Avenue provides 21 MATLAB GT files in some distributions while 22 ROI test folders may exist. Missing GT is excluded, never relabeled zero. GT length must match raw source-frame count.
5. Dataset acquisition and third-party code/weights are user-supplied: do not redistribute without permission.
6. Inference model is per-frame ROI-based. No temporal model is implemented despite narrative descriptions of motion cues.

**Reproduced published numbers? NO.** New runs must report metric differences and precise protocol.
