"""Compare newly computed values with published 2025 Table II, without implying reproduction."""
import argparse
import csv
import json
from pathlib import Path

PUBLISHED={
    'Ped1':dict(precision=.6998,recall=.3761,f1=.4893,auc_roc=.5324),
    'Ped2':dict(precision=.9305,recall=.4934,f1=.6449,auc_roc=.5476),
    'Avenue':dict(precision=.1956,recall=.1428,f1=.1651,auc_roc=.5391),
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',default='results')
    args=p.parse_args()
    root=Path(args.results)
    rows=[]
    for ds,ref in PUBLISHED.items():
        path=root/ds/'evaluation_summary.json'
        if not path.is_file():
            print(f'[MISSING RESULT] {ds}: {path}');continue
        payload=json.loads(path.read_text(encoding='utf-8'))
        test=payload['test']
        for name,old in ref.items():
            new=float(test[name])
            rows.append(dict(dataset=ds,metric=name,published=old,reconstructed=new,
                             absolute_difference=round(abs(new-old),6),
                             publication_reproduced='NOT VERIFIED'))
            print(f'{ds:<7} {name:<10} published={old:.4f}, new={new:.4f}, abs_delta={abs(new-old):.4f}')
    if rows:
        out=root/'comparison_with_paper.csv';out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
        print('Saved:',out)
    else:
        print('No new experiment runs found. Published numbers are reference only.')


if __name__=='__main__':main()
