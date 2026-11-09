"""Frame-level evaluation for UCSD Ped1, UCSD Ped2 and CUHK Avenue.

Calibration uses validation-only scores. Missing ground truth is excluded.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roi_vad.model import load_checkpoint
from roi_vad.evaluation import (predict_frame_scores,assemble_labeled_frames,
                split_sequences,select_threshold,metrics,write_records,save_plots)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['Ped1','Ped2','Avenue'],required=True)
    p.add_argument('--processed-test',required=True,help='Folder with manifests and rois')
    p.add_argument('--gt-root',required=True,help='UCSD Test_gt or Avenue testing_label_mask')
    p.add_argument('--checkpoint',required=True,help='Trained AE .pth (not detector)')
    p.add_argument('--result-dir',required=True)
    p.add_argument('--threshold-mode',choices=['f1','p95','meanstd'],required=True,
                   help='Choose explicitly to resolve alternative published threshold descriptions')
    p.add_argument('--k',type=float,default=3.,help='Mean+k*std threshold only')
    p.add_argument('--no-roi',choices=['skip','zero'],default='skip')
    p.add_argument('--val-ratio',type=float,default=0.30)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    model=load_checkpoint(args.checkpoint,args.device)
    scored=predict_frame_scores(model,args.processed_test,args.batch_size,args.device)
    records,coverage=assemble_labeled_frames(args.dataset,args.processed_test,
                                        args.gt_root,scored,args.no_roi)
    val_rows,test_rows,split=split_sequences(records,args.val_ratio,args.seed)
    vs=np.asarray([r['score'] for r in val_rows]);vy=np.asarray([r['label'] for r in val_rows])
    threshold,sweep=select_threshold(vy,vs,args.threshold_mode,args.k)
    result_dir=Path(args.result_dir);result_dir.mkdir(parents=True,exist_ok=True)
    summary={'dataset':args.dataset,'implementation':'reference implementation',
             'threshold_mode':args.threshold_mode,'frame_pooling':'max ROI MSE',
             'no_roi':args.no_roi,'threshold_k':args.k,
             'validation':metrics(val_rows,threshold),'test':metrics(test_rows,threshold),
             'split':split,'coverage':coverage}
    write_records(result_dir/'validation_frames.csv',val_rows,threshold)
    write_records(result_dir/'test_frames.csv',test_rows,threshold)
    if sweep:
        with (result_dir/'validation_threshold_sweep.csv').open('w',newline='',encoding='utf-8') as fh:
            w=csv.DictWriter(fh,fieldnames=['threshold','f1','precision','recall'])
            w.writeheader();w.writerows(sweep)
    np.save(result_dir/'scores.npy',np.asarray([r['score'] for r in test_rows]))
    np.save(result_dir/'labels.npy',np.asarray([r['label'] for r in test_rows],dtype=np.int8))
    (result_dir/'evaluation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    save_plots(test_rows,result_dir,args.dataset)
    print(json.dumps(summary,indent=2))
    print(f'Wrote {result_dir} (frame-level, validation threshold locked).')

if __name__=='__main__': main()
