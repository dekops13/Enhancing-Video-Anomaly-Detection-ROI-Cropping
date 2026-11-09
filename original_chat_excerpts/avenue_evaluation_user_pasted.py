# SOURCE: Complete code cell pasted directly by user; additional comments here mark provenance.
# STATUS: CHAT-VERBATIM (excluding these header comments); associated output 20879 ROI samples and 3544 anomaly labels.
# CAUTION: assigns GT by ROI file order and uses test-score P95; not verified frame-level benchmark.
# ===== Evaluasi Avenue dengan GT PNG =====
import torch, cv2, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (32, 32)

AVENUE_ROOT = Path(r"C:\Users\dekops\miniconda3\Dataset\Avenue")
CROPS_ROOT = AVENUE_ROOT / "Yolo5LocalOutput" / "crops" / "Test"
GT_PNG_ROOT = AVENUE_ROOT / "testing_label_mask"

model_path = "Avenruns/roi_ae32/avenue_best_model.pth"

# --- Arsitektur ConvAE (sesuai training augmentasi) ---
class ConvAE(torch.nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1), torch.nn.ReLU(),
        )
        self.fc1 = torch.nn.Linear(128*4*4, latent_dim)
        self.fc2 = torch.nn.Linear(latent_dim, 128*4*4)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), torch.nn.Sigmoid(),
        )
    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        z = self.fc1(z)
        z = self.fc2(z)
        z = z.view(z.size(0), 128, 4, 4)
        return self.decoder(z)

# --- Load model ---
ckpt = torch.load(model_path, map_location=DEVICE)
model = ConvAE()
if isinstance(ckpt, dict) and "model" in ckpt:
    model.load_state_dict(ckpt["model"], strict=False)
else:
    model.load_state_dict(ckpt, strict=False)
model.to(DEVICE).eval()
print(f"✅ Model loaded: {model_path}")

# --- Loader GT Avenue ---
def load_gt_labels(seq_name: str, n_frames: int):
    folder = GT_PNG_ROOT / seq_name
    labels = np.zeros(n_frames, dtype=np.uint8)
    for i in range(1, n_frames+1):
        fpath = folder / f"{i:04d}.png"
        if fpath.exists():
            im = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
            labels[i-1] = 1 if cv2.countNonZero(im) > 0 else 0
    return labels

# --- Evaluasi ---
all_scores, all_labels = [], []
seqs = sorted([p for p in CROPS_ROOT.iterdir() if p.is_dir()])
print(f"[CHECK] Jumlah sequence ROI Test: {len(seqs)}")

for sd in seqs:
    files = sorted([p for p in sd.iterdir() if p.is_file()])
    n_frames = len(files)
    gt = load_gt_labels(sd.name, n_frames)

    for i, fp in enumerate(files):
        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE).astype(np.float32) / 255.0
        x = torch.from_numpy(img)[None,None].to(DEVICE)
        with torch.no_grad():
            rec = model(x)
            mse = torch.nn.functional.mse_loss(rec, x, reduction="mean").item()
        all_scores.append(mse)
        all_labels.append(gt[i] if i < len(gt) else 0)

all_scores = np.array(all_scores)
all_labels = np.array(all_labels)
print(f"[SUMMARY] Total frames={len(all_scores)} | anomali={all_labels.sum()}")

# Threshold dari val set → pakai quantile
thr = np.percentile(all_scores, 95)
preds = (all_scores > thr).astype(int)

auc = roc_auc_score(all_labels, all_scores) if all_labels.sum()>0 else float("nan")
P,R,F1,_ = precision_recall_fscore_support(all_labels, preds, average="binary", zero_division=0)
cm = confusion_matrix(all_labels, preds)

print(f"[TEST] Thr={thr:.6f} | P={P:.4f} | R={R:.4f} | F1={F1:.4f} | AUC={auc:.4f}")
print("CM [[TN FP] [FN TP]] =")
print(cm)
