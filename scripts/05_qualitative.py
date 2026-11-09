"""Create a figure of *verified* TP/FP/FN from test_frames.csv (frame-level).

Only use the source_path in per-frame manifest. Never reconstruct an image path
by flattening ROI folders in an arbitrary order.
"""
import argparse
import csv
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-dir',required=True)
    p.add_argument('--output',default=None)
    a=p.parse_args()
    path=Path(a.result_dir)
    with (path/'test_frames.csv').open(newline='',encoding='utf-8') as fh:
        rows=list(csv.DictReader(fh))
    cases={case:next((r for r in rows if r['case']==case and r['source_path']
                      and Path(r['source_path']).is_file()),None)
           for case in ('TP','FP','FN')}
    if not all(cases.values()):
        missing=[c for c,r in cases.items() if r is None]
        raise ValueError('No verifiable original source-frame paths for: '+', '.join(missing)+
                         '. Reprocess image sequences, retain originals, or explicitly add frame extraction.')
    fig,axes=plt.subplots(1,3,figsize=(11,3.7))
    captions={'TP':'True Positive','FP':'False Positive','FN':'False Negative'}
    for ax,(case,row) in zip(axes,cases.items()):
        im=cv2.imread(row['source_path'],cv2.IMREAD_COLOR)
        if im is None:raise ValueError('Cannot read frame: '+row['source_path'])
        ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        ax.set_title(f'{case} — {captions[case]}\n'+
                     f"{row['sequence']} frame {row['frame_index']}, MSE={float(row['score']):.5f}")
        ax.axis('off')
    fig.tight_layout()
    output=Path(a.output) if a.output else path/'qualitative_TP_FP_FN.png'
    fig.savefig(output,dpi=300,bbox_inches='tight');plt.close(fig)
    print('Saved',output)


if __name__=='__main__':main()
