# Source recovery after local deletion (2026-11-09)

## Status
Partial chat recovery. The user confirmed the local original source was deleted. No verified original Python/Jupyter files or checkpoints located on connected Google Drive, in the Library, or in the three GitHub repositories visible to the connected account. **Do not present these scripts as a full reproducible implementation.**

## Materials present
- `original_chat_excerpts/ucsd_validation_user_pasted.py`: complete UCSD Ped1 validation cell pasted by the user. Original code was configured with `models\best_model.pth`; not guaranteed to be the identical run using `best_modelped1aug.pth`.
- `original_chat_excerpts/avenue_evaluation_user_pasted.py`: complete Avenue evaluation cell pasted by the user, matching narrative of 20,879 ROI scores and 3,544 positives. This is not a validated frame-level evaluator.
- `source_from_chat/cnn_autoencoder.py`: user-supplied network class verbatim.
- `source_from_chat/ucsd_gt_loader.py`: user-supplied GT parsing functions verbatim.
- `docs/PROVENANCE.md`: executed logs and discrepancies.

## Critical missing artifacts
- Actual YOLOv5 + BSMOG2 + ROI cropping source and detector weights.
- AE training + augmentation source and immutable per-dataset split manifests.
- `models/best_modelped1aug.pth`, Ped2 and Avenue AE checkpoint binaries.
- Raw ROI images, full-frame index mappings, valid/test score files.
- No reconstruction of these items is included.

## Critical scientific cautions
1. Avenue user-pasted evaluator sets `n_frames=len(ROI files)` and assigns mask label `gt[i]`, with no actual ROI filename to source-frame mapping. Multiple objects per source frame can silently mislabel samples.
2. Avenue user-pasted evaluator says `threshold from val` but actually uses the P95 of ALL TEST SCORES, so it leaks test statistics into threshold tuning.
3. UCSD Ped1 GT folder has only 10 matching subset sequences of 36 ROI test sequence folders in the later logged experiment. Unmatched sequences must be excluded from labeled metrics, not assumed normal.
4. The actual last logged matched-GT Ped1 run used 3 labeled validation sequences/7 labeled test sequences, threshold 0.002526 and per-ROI metrics (P 0.4657, R 0.3687, F1 0.4116, AUC 0.4952). The assistant-drafted matched-GT script is not included as original user source.
5. Paper PDF reports different metrics and 95th-percentile validation threshold. Don't claim their equality, or call any per-ROI scores frame-level before source-frame aggregation is verified.
6. TP/FP/FN collage script previously used mean+3std threshold and flattened files across all sequences without respecting filtered subsets. Such plots cannot be trusted as provenance-safe examples.
7. `torch.load(..., strict=False)` can hide missing/unexpected parameter keys; training definition and weights must be verified.

## Connected source audit
- Connected Google Drive search: papers present; original `.ipynb`, `.py`, `.pth` not located.
- Connected GitHub: `dekops13/myproject`, `dekops13/projectKAS`, `dekops13/SentiKampus-Course-v2` visible; target repository not present. No verified VAD source discovered.
- Files Library: publication PDF, but unrelated deepfake FF++ notebooks must not be included as VAD source.

## Recovery opportunities outside the connected services
Recycle Bin; OneDrive/Windows File History/backup and synced directories; Jupyter `.ipynb_checkpoints`; VS Code local history; Miniconda notebook checkpoints; external drive/old ZIP/email attachment. Avoid writing new data to the source disk if filesystem-level recovery is attempted.
