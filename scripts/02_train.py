"""RECONSTRUCTED normal-only CNN-AE training on preprocessed 32x32 ROIs.

Paper: 50 epochs maximum, batch=128, Adam, MSE, early stopping. Learning rate,
augmentations, seed and patience were not recoverable; configurable defaults.
"""
import argparse
import json
import random
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roi_vad.model import CNNAutoencoder


class ROIImages(Dataset):
    def __init__(self, paths, augment=False):
        self.paths, self.augment = list(paths), augment
    def __len__(self): return len(self.paths)
    def __getitem__(self, ix):
        with Image.open(self.paths[ix]) as im:
            a = np.array(im.convert('L').resize((32, 32)), dtype=np.float32).copy()/255.
        if self.augment:
            # Optional RECONSTRUCTED augmentation; not verified in original experiment.
            if random.random() < 0.5: a = np.fliplr(a).copy()
        return torch.from_numpy(a.copy())[None, ...]


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    root = Path(args.rois)
    sequences = sorted([p for p in root.iterdir() if p.is_dir()])
    if len(sequences) < 2:
        raise RuntimeError('Training requires at least two sequence folders for video-level holdout.')
    rng = random.Random(args.seed)
    rng.shuffle(sequences)
    k = max(1, min(len(sequences)-1, round(len(sequences)*args.val_ratio)))
    val_seq, train_seq = sequences[:k], sequences[k:]
    def collect(seqs):
        return [p for seq in seqs for p in sorted(seq.iterdir()) if p.suffix.lower() in ('.png','.jpg','.jpeg')]
    train_files, val_files = collect(train_seq), collect(val_seq)
    if not train_files or not val_files:
        raise RuntimeError('Empty train/validation ROI split; check preprocessing output.')
    tr = DataLoader(ROIImages(train_files,args.augment), batch_size=args.batch_size, shuffle=True, num_workers=0)
    va = DataLoader(ROIImages(val_files),batch_size=args.batch_size,shuffle=False,num_workers=0)
    device = torch.device(args.device)
    model = CNNAutoencoder().to(device)
    optim = torch.optim.Adam(model.parameters(),lr=args.lr)
    best = float('inf'); patience = 0; history=[]
    out = Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    metadata = {'provenance':'RECONSTRUCTED, not recovered original training script',
            'train_sequences':[s.name for s in train_seq],
            'val_sequences':[s.name for s in val_seq],
            'arguments':vars(args)}
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for x in tr:
            x=x.to(device); optim.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(model(x),x)
            loss.backward();optim.step();losses.append(float(loss.item())*len(x))
        model.eval(); total=0.;n=0
        with torch.no_grad():
            for x in va:
                x=x.to(device);v=torch.nn.functional.mse_loss(model(x),x,reduction='sum')
                total+=float(v.item());n+=x.numel()
        vloss=total/n
        row={'epoch':epoch,'train_mse':sum(losses)/len(train_files),'validation_pixel_mse':vloss}
        history.append(row)
        print(f"Epoch {epoch:03d}: train={row['train_mse']:.6f}, val={vloss:.6f}")
        if vloss < best-args.min_delta:
            best=vloss;patience=0
            torch.save({'model':model.state_dict(),'epoch':epoch,'best_val_mse':best,
                        'config':metadata},out)
        else:
            patience+=1
            if patience >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break
    (out.parent/(out.stem+'_training.json')).write_text(json.dumps({'metadata':metadata,
        'history':history,'best_validation_mse':best},indent=2),encoding='utf-8')
    print(f'Saved best reconstructed checkpoint: {out}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rois',required=True,help='data/processed/<dataset>/train/rois')
    p.add_argument('--output',required=True,help='e.g., checkpoints/ped1/reconstructed_best.pth')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=1e-3,help='Unverified reconstruction default')
    p.add_argument('--patience',type=int,default=7,help='Unverified reconstruction default')
    p.add_argument('--val-ratio',type=float,default=0.20)
    p.add_argument('--min-delta',type=float,default=1e-6)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--augment',action='store_true',help='Unverified optional horizontal flips')
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if not 0 < args.val_ratio < 1: p.error('--val-ratio must be in (0,1)')
    train(args)

if __name__=='__main__': main()
