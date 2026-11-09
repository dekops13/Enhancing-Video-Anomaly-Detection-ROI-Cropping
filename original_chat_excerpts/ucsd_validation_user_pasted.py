# SOURCE: Complete code cell pasted directly by user in prior research conversation.
# STATUS: CHAT-VERBATIM (excluding this header); exact executed checkpoint/config is unverified.
# NOTE: MODEL_PATH from source was models\best_model.pth, not best_modelped1aug.pth.
# ===== ROI Autoencoder 32x32 — VALIDATION (Jupyter) =====
import os, re, json, numpy as np, cv2, torch
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from PIL import Image
from tqdm import tqdm

# ---------- CONFIG (ubah sesuai lokasi Anda) ----------
UCSD_ROOT     = r"C:\Users\dekops\miniconda3\Dataset\UCSDped1"                 # berisi Train, Test, Test_gt
CROPS_ROOT    = r"C:\Users\dekops\miniconda3\Dataset\UCSDped1\Yolo5LocalOutput\crops"  # berisi Train, Test\Seq\*.png
MODEL_PATH    = r"models\best_model.pth"                                    # model 5-epoch yg baru disimpan
IMG_SIZE      = (32, 32)
LATENT_DIM    = 128

# subset validasi diambil dari sebagian sequence pada Test (mis. 30% pertama secara alfabet)
VAL_RATIO     = 0.30

DEVICE = torch.device("cpu")  # konsisten dengan training Anda di CPU
# -------------------------------------------------------

# ---------- Model (harus sama dengan saat training) ----------
class CNNAutoencoder(torch.nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1,32,3,2,1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32,64,3,2,1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64,128,3,2,1), torch.nn.ReLU(inplace=True),
        )
        self.fc1 = torch.nn.Linear(128*4*4, latent_dim)
        self.fc2 = torch.nn.Linear(latent_dim, 128*4*4)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128,64,3,2,1,1), torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(64,32,3,2,1,1), torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32,1, 3,2,1,1), torch.nn.Sigmoid(),
        )
    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        h = self.fc1(z)
        z2 = self.fc2(h).view(-1,128,4,4)
        return self.decoder(z2)

# ---------- Util ----------
def list_seq_dirs(p: Path):
    if not p.exists(): return []
    return sorted([d for d in p.iterdir() if d.is_dir()])

def parse_frame_index(filename: str):
    # cocokkan pola "_f00012" → 12
    m = re.search(r"_f(\d+)", filename)
    return int(m.group(1)) if m else None

def gt_labels_for_seq(seq_name: str):
    """Buat label per-frame untuk satu sequence berdasarkan mask Test_gt\<seq>_gt\*.
       Label 1 jika ada piksel >0, else 0. """
    gt_dir = Path(UCSD_ROOT) / "Test_gt" / f"{seq_name}_gt"
    if not gt_dir.exists(): return []
    masks = sorted([p for p in gt_dir.iterdir() if p.is_file()])
    labels = []
    for m in masks:
        im = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        labels.append(1 if (im is not None and cv2.countNonZero(im) > 0) else 0)
    return labels  # len = jumlah frame GT

@torch.no_grad()
def scores_and_labels_for_seq(model, seq_dir: Path, seq_name: str):
    """Hitung MSE per-ROI + ambil label per-ROI dari GT frame (berdasarkan index frame di nama file)."""
    files = sorted([p for p in seq_dir.iterdir() if p.is_file()])
    gt = gt_labels_for_seq(seq_name)
    out_scores, out_labels = [], []
    for fp in files:
        fidx = parse_frame_index(fp.name)
        if fidx is None:  # skip ROI yang tidak punya indeks frame
            continue
        # map ke gt index (GT biasanya 1-based atau 0-based bergantung dataset;
        # UCSD mask biasanya sama urutan nama → pakai fidx-1 jika nama ROI mulai dari 1)
        gi = fidx - 1 if fidx > 0 and fidx-1 < len(gt) else (fidx if fidx < len(gt) else None)
        if gi is None or gi >= len(gt):  # tak ada GT untuk frame ini
            continue
        # read ROI grayscale
        img = Image.open(fp).convert("L").resize(IMG_SIZE, Image.BILINEAR)
        x = torch.from_numpy(np.array(img, dtype=np.float32)/255.0)[None,None,...].to(DEVICE)
        rec = model(x)
        mse = torch.nn.functional.mse_loss(rec, x, reduction="mean").item()
        out_scores.append(mse)
        out_labels.append(int(gt[gi]))
    return np.array(out_scores, np.float32), np.array(out_labels, np.int32)

def concat_split(d):
    scores, labels = [], []
    for s,l in d:
        scores.append(s); labels.append(l)
    if len(scores)==0: return np.array([],np.float32), np.array([],np.int32)
    return np.concatenate(scores), np.concatenate(labels)

def metrics_at_threshold(y_true, y_score, thr):
    pred = (y_score > thr).astype(np.int32)
    if pred.sum()==0 and y_true.sum()==0:
        p=r=f=f1=1.0
    else:
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f1= f1_score(y_true, pred, zero_division=0)
    return p,r,f1

# ---------- Load model ----------
model = CNNAutoencoder(LATENT_DIM).to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
# dukung checkpoint (dict) atau state_dict murni
if isinstance(state, dict) and "model" in state:
    model.load_state_dict(state["model"], strict=True)
else:
    model.load_state_dict(state, strict=True)
model.eval()
print(f"✅ Model loaded: {MODEL_PATH}")

# ---------- Kumpulkan sequence Test, bagi menjadi VAL & TEST ----------
test_seq_root = Path(CROPS_ROOT) / "Test"
seq_dirs = list_seq_dirs(test_seq_root)
if not seq_dirs:
    raise RuntimeError(f"Tidak menemukan ROI Test di {test_seq_root}")

n_val = max(1, int(len(seq_dirs)*VAL_RATIO))
val_seqs  = seq_dirs[:n_val]
test_seqs = seq_dirs[n_val:]
print(f"VAL seqs: {len(val_seqs)} | TEST seqs: {len(test_seqs)}")

# ---------- Hitung skor+label di VAL ----------
val_pairs = []
for sd in tqdm(val_seqs, desc="VAL seqs"):
    s,l = scores_and_labels_for_seq(model, sd, sd.name)
    if len(s)>0: val_pairs.append((s,l))
val_scores, val_true = concat_split(val_pairs)
print("VAL samples(per-ROI):", len(val_scores), "| anomali:", int(val_true.sum()))

if len(val_scores)==0:
    raise RuntimeError("VAL kosong—cek path CROPS_ROOT/Test atau pola nama file ROI (_fXXXXX).")

# pilih threshold via grid percentile dari distribusi val
percentiles = np.linspace(50, 99.5, 100)
cands = np.unique(np.percentile(val_scores, percentiles))
best = {"thr": None, "f1": -1, "prec":0, "rec":0}
for thr in cands:
    p,r,f = metrics_at_threshold(val_true, val_scores, thr)
    if (f > best["f1"]) or (f == best["f1"] and r > best["rec"]) or (f==best["f1"] and r==best["rec"] and thr < (best["thr"] or thr)):
        best = {"thr": float(thr), "f1": float(f), "prec": float(p), "rec": float(r)}

val_auc = roc_auc_score(val_true, val_scores) if len(np.unique(val_true))>1 else float('nan')
print(f"[VAL] BestThr={best['thr']:.6f} | F1={best['f1']:.4f} | P={best['prec']:.4f} | R={best['rec']:.4f} | AUC={val_auc:.4f}")

# ---------- Hitung skor+label di TEST, pakai thr dari VAL ----------
test_pairs = []
for sd in tqdm(test_seqs, desc="TEST seqs"):
    s,l = scores_and_labels_for_seq(model, sd, sd.name)
    if len(s)>0: test_pairs.append((s,l))
test_scores, test_true = concat_split(test_pairs)
print("TEST samples(per-ROI):", len(test_scores), "| anomali:", int(test_true.sum()))

thr = best["thr"]
test_pred = (test_scores > thr).astype(np.int32)

test_prec = precision_score(test_true, test_pred, zero_division=0)
test_rec  = recall_score(test_true, test_pred, zero_division=0)
test_f1   = f1_score(test_true, test_pred, zero_division=0)
test_auc  = roc_auc_score(test_true, test_scores) if len(np.unique(test_true))>1 else float('nan')
cm = confusion_matrix(test_true, test_pred)

# ringkasan normal vs anomali
val_norm = len(val_true) - int(val_true.sum())
test_norm = len(test_true) - int(test_true.sum())

print(f"[TEST] Thr={thr:.6f} | P={test_prec:.4f} | R={test_rec:.4f} | F1={test_f1:.4f} | AUC={test_auc:.4f}")
print("Confusion matrix [ [TN FP] [FN TP] ]:")
print(cm)
print(f"VAL: normal={val_norm} | anomaly={int(val_true.sum())}")
print(f"TEST: normal={test_norm} | anomaly={int(test_true.sum())}")
