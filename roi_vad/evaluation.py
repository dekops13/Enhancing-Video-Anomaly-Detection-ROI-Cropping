"""Frame-level anomaly scoring, sequence split and validation-only thresholding.

NEW RECONSTRUCTION. The user's earlier snippets sometimes evaluated per ROI or used
P95 of test scores. This implementation instead evaluates source frames, checks GT
length, records no-ROI coverage, and never selects a threshold using test scores.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score,
)
from .gt import load_labels


def read_manifest(path):
    with Path(path).open(newline='',encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def resolve_roi(row, test_root):
    path = Path(row['roi_path'])
    if path.is_file(): return path
    fallback = test_root/'rois'/row['sequence']/path.name
    if fallback.is_file(): return fallback
    raise FileNotFoundError(f'ROI in manifest missing: {path} / {fallback}')


def predict_frame_scores(model, test_root, batch_size=128, device='cpu'):
    """Return {(sequence,1-based-frame-index): maximum per-ROI MSE}.

    Max pooling is a declared reconstruction choice: original pooling rule unknown.
    """
    test_root = Path(test_root)
    rows = read_manifest(test_root/'roi_manifest.csv')
    scores = defaultdict(list)
    model.eval()
    for start in range(0,len(rows),batch_size):
        chunk=rows[start:start+batch_size]
        images=[]
        for row in chunk:
            with Image.open(resolve_roi(row,test_root)) as im:
                a=np.asarray(im.convert('L').resize((32,32)),dtype=np.float32)/255.
            images.append(a)
        x=torch.from_numpy(np.stack(images)[:,None]).to(device)
        with torch.no_grad():
            pred=model(x)
            losses=(pred-x).square().mean(dim=(1,2,3)).detach().cpu().numpy()
        for row,loss in zip(chunk,losses):
            scores[row['sequence'],int(row['frame_index'])].append(float(loss))
    return {key:max(vals) for key,vals in scores.items()}


def assemble_labeled_frames(dataset, test_root, gt_root, scored, no_roi='skip'):
    """Build independent source-frame labels; never fabricate missing GT.

    If no_roi='zero', absent ROI gets anomaly score zero (explicit experiment
    assumption). Default='skip' omits uncovered frames and reports coverage.
    """
    test_root=Path(test_root)
    frames=read_manifest(test_root/'frame_manifest.csv')
    by_seq=defaultdict(list)
    for row in frames:
        by_seq[row['sequence']].append(row)
    records=[];excluded=[];missing_roi=0
    for seq, items in sorted(by_seq.items()):
        gt=load_labels(dataset,gt_root,seq)
        if gt is None:
            excluded.append(seq)
            continue
        # Strict check prevents accidental label shifting caused by ROI count != frames.
        frame_idx=[int(row['frame_index']) for row in items]
        if len(gt)!=len(items) or sorted(frame_idx)!=list(range(1,len(items)+1)):
            raise ValueError(f'GT/frame mismatch {seq}: GT={len(gt)}, raw frames={len(items)}; '
                'inspect extraction frame rate / mask alignment before proceeding')
        for row in items:
            i=int(row['frame_index'])
            key=(seq,i)
            if key not in scored:
                missing_roi+=1
                if no_roi=='skip': continue
                if no_roi!='zero': raise ValueError(f'Unknown no_roi option {no_roi}')
            records.append(dict(sequence=seq,frame_index=i,label=int(gt[i-1]),
                    score=float(scored.get(key,0.)),num_rois=int(row['num_rois']),
                    source_path=row['source_path']))
    return records,{'unlabeled_sequences_excluded':excluded,
                    'annotated_source_frames':sum(len(items) for seq,items in by_seq.items() if seq not in excluded),
                    'frames_with_no_roi':missing_roi,'frames_evaluated':len(records),
                    'coverage':len(records)/max(1,sum(len(items) for seq,items in by_seq.items() if seq not in excluded))}


def split_sequences(records, val_ratio=0.3, seed=42):
    """Reproducible, disjoint sequence-level split with positive/negative support.

    This is a *reconstructed* stratified split; split of published run unknown.
    The procedure never optimizes F1/AUC using test labels.
    """
    by_seq=defaultdict(list)
    for row in records:
        by_seq[row['sequence']].append(row)
    seqs=sorted(by_seq)
    if len(seqs)<2:
        raise ValueError('Need at least two annotated sequences')
    n_val=max(1,min(len(seqs)-1,round(val_ratio*len(seqs))))
    rng=np.random.default_rng(seed)
    chosen=None
    for _ in range(10000):
        val=set(rng.choice(seqs,n_val,replace=False).tolist())
        val_labels={r['label'] for seq in val for r in by_seq[seq]}
        test_labels={r['label'] for seq in seqs if seq not in val for r in by_seq[seq]}
        if val_labels=={0,1} and test_labels=={0,1}:
            chosen=val;break
    if chosen is None:
        raise ValueError('No valid split found after 10000 deterministic trials; '
                         'verify GT coverage or supply a manual sequence partition.')
    return ([r for r in records if r['sequence'] in chosen],
            [r for r in records if r['sequence'] not in chosen],
            {'seed':seed,'val_sequences':sorted(chosen),
             'test_sequences':sorted(set(seqs)-chosen),'val_ratio':val_ratio,
             'provenance':'RECONSTRUCTED stratified sequence split'})


def select_threshold(labels,scores,mode='f1',k=3.0):
    """Pick threshold on *validation only*; exact original protocol unresolved.

    O(n log n) F1 sweep over all unique scores, not an approximation grid.
    """
    labels=np.asarray(labels,dtype=int)
    scores=np.asarray(scores,dtype=float)
    if scores.size==0 or labels.shape!=scores.shape or not np.all(np.isfinite(scores)):
        raise ValueError('Invalid or empty validation labels/scores')
    if mode=='p95':
        return float(np.percentile(scores,95)),[]
    if mode=='meanstd':
        return float(scores.mean()+k*scores.std()),[]
    if mode!='f1':
        raise ValueError(f'Unknown mode: {mode}')
    if set(labels)!={0,1}:
        raise ValueError('F1-optimal threshold requires both classes in validation')
    order=np.argsort(-scores,kind='stable')
    sorted_scores=scores[order]
    sorted_labels=labels[order]
    positive=int(labels.sum())
    tp=0;fp=0;audit=[]
    # A threshold equal to a unique score excludes all samples with that score
    # (prediction uses score > threshold). Sweep from +infinity downward.
    audit.append({'threshold':float(sorted_scores[0]),'f1':0.,'precision':0.,'recall':0.})
    start=0
    while start<len(scores):
        end=start+1
        while end<len(scores) and sorted_scores[end]==sorted_scores[start]:
            end+=1
        tp+=int(sorted_labels[start:end].sum())
        fp+=end-start-int(sorted_labels[start:end].sum())
        if end<len(scores):
            threshold=float(sorted_scores[end])
            # all samples with score greater than threshold have now been included
            # but scores equal to threshold should still be excluded
            if end<len(scores) and sorted_scores[end]==threshold:
                # threshold at next distinct value; next group excluded
                pass
            precision=tp/(tp+fp) if tp+fp else 0.
            recall=tp/positive
            f1=2*tp/(2*tp+fp+(positive-tp)) if 2*tp+fp+(positive-tp) else 0.
            audit.append({'threshold':threshold,'f1':f1,'precision':precision,'recall':recall})
        else:
            # Under score>threshold, below-min threshold includes every ROI
            threshold=float(np.nextafter(sorted_scores[-1],-np.inf))
            precision=tp/(tp+fp)
            recall=tp/positive
            f1=2*tp/(2*tp+fp+positive-tp)
            audit.append({'threshold':threshold,'f1':f1,'precision':precision,'recall':recall})
        start=end
    best=max(audit,key=lambda r:(r['f1'],r['recall'],-r['threshold']))
    return float(best['threshold']),audit


def metrics(rows,thr):
    y=np.asarray([r['label'] for r in rows],dtype=np.int8)
    s=np.asarray([r['score'] for r in rows],dtype=np.float64)
    if len(y)==0 or set(y)!={0,1}:
        raise ValueError('Both normal and anomaly labels needed for ROC-PR evaluation')
    pred=(s>thr).astype(np.int8)
    cm=confusion_matrix(y,pred,labels=[0,1])
    return {'samples':int(len(y)),'anomaly_frames':int(y.sum()),
            'normal_frames':int(len(y)-y.sum()),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),
            'f1':float(f1_score(y,pred,zero_division=0)),
            'auc_roc':float(roc_auc_score(y,s)),
            'average_precision':float(average_precision_score(y,s)),
            'confusion_matrix':cm.tolist(),'threshold':float(thr)}


def write_records(path,rows,threshold):
    fields=['sequence','frame_index','source_path','num_rois','label','score','prediction','case']
    with Path(path).open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=fields);writer.writeheader()
        for r in rows:
            d=dict(r);d['prediction']=int(r['score']>threshold)
            d['case']={ (1,1):'TP',(0,1):'FP',(1,0):'FN',(0,0):'TN'}[(r['label'],d['prediction'])]
            writer.writerow(d)


def save_plots(rows,output,caption):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    y=np.asarray([r['label'] for r in rows]);s=np.asarray([r['score'] for r in rows])
    fpr,tpr,_=roc_curve(y,s)
    prec,rec,_=precision_recall_curve(y,s)
    fig,ax=plt.subplots(figsize=(5.5,4.3))
    ax.plot(fpr,tpr,label=f'ROC AUC={roc_auc_score(y,s):.4f}')
    ax.plot([0,1],[0,1],'--',alpha=0.5)
    ax.set(xlabel='False Positive Rate',ylabel='True Positive Rate',title=f'{caption} – ROC')
    ax.legend();fig.tight_layout();fig.savefig(output/'roc_curve.png',dpi=300);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.5,4.3))
    ax.plot(rec,prec,label=f'AP={average_precision_score(y,s):.4f}')
    ax.set(xlabel='Recall',ylabel='Precision',title=f'{caption} – Precision–Recall')
    ax.legend();fig.tight_layout();fig.savefig(output/'pr_curve.png',dpi=300);plt.close(fig)
