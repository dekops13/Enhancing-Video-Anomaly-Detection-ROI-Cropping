"""Unit tests on synthetic inputs; published benchmark values are not evaluated."""
import csv
from pathlib import Path
import numpy as np
import cv2
import torch
from scipy.io import savemat
from roi_vad.model import CNNAutoencoder, load_checkpoint
from roi_vad.gt import load_labels
from roi_vad.io import letterbox_grayscale
from roi_vad.evaluation import select_threshold, split_sequences, assemble_labeled_frames, metrics


def test_model_shapes_and_checkpoint(tmp_path):
    model=CNNAutoencoder()
    x=torch.rand(2,1,32,32)
    assert model.encoder(x).shape==(2,128,4,4)
    with torch.no_grad():
        assert model(x).shape==x.shape
    path=tmp_path/'checkpoint.pth'
    torch.save({'model':model.state_dict()},path)
    loaded=load_checkpoint(path)
    assert torch.allclose(loaded.fc1.weight,model.fc1.weight)


def test_ground_truth_ucsd_and_avenue_mat(tmp_path):
    uc=tmp_path/'uc';(uc/'Test003_gt').mkdir(parents=True)
    for i, flag in enumerate([0,1,0],start=1):
        im=np.zeros((18,30),np.uint8)
        if flag: im[5:10,5:10]=255
        cv2.imwrite(str(uc/'Test003_gt'/f'{i:03d}.tif'),im)
    assert load_labels('Ped1',uc,'Test003').tolist()==[0,1,0]
    assert load_labels('Ped1',uc,'Test004') is None
    mat=tmp_path/'mat';mat.mkdir()
    cells=np.empty((1,3),dtype=object)
    for i, flag in enumerate([0,1,0]):
        mask=np.zeros((20,30),np.uint8)
        if flag: mask[3,4]=1
        cells[0,i]=mask
    savemat(mat/'1_label.mat',{'volLabel':cells})
    assert load_labels('Avenue',mat,'01').tolist()==[0,1,0]
    assert load_labels('Avenue',mat,'22') is None


def test_threshold_is_validation_only():
    labels=np.asarray([0,0,1,1])
    scores=np.asarray([.1,.2,.7,.8])
    thr, sweep=select_threshold(labels,scores,'f1')
    assert len(sweep)>1
    assert np.array_equal(scores>thr,[0,0,1,1])
    p95,_=select_threshold(labels,scores,'p95')
    assert p95==np.percentile(scores,95)
    ms,_=select_threshold(labels,scores,'meanstd',k=3)
    assert np.isclose(ms,scores.mean()+3*scores.std())


def test_metrics_and_sequence_split():
    records=[]
    for seq in ('Test003','Test004','Test014','Test018'):
        for index, score, label in ((1,.1,0),(2,.8,1)):
            records.append({'sequence':seq,'frame_index':index,'score':score,'label':label})
    v,t,info=split_sequences(records,val_ratio=.5,seed=42)
    assert set(info['val_sequences']).isdisjoint(info['test_sequences'])
    thr,_=select_threshold([r['label'] for r in v],[r['score'] for r in v],'f1')
    result=metrics(t,thr)
    assert result['f1']==1 and result['auc_roc']==1


def test_unannotated_and_missing_roi_do_not_get_fabricated_labels(tmp_path):
    proc=tmp_path/'proc';proc.mkdir()
    gt=tmp_path/'gt';(gt/'Test003_gt').mkdir(parents=True)
    for i,lab in enumerate([0,1],start=1):
        image=np.zeros((18,18),dtype=np.uint8)
        if lab: image[2:5,2:5]=255
        cv2.imwrite(str(gt/'Test003_gt'/f'{i:03d}.png'),image)
    rows=[]
    for seq in ('Test003','Test001'):
        for i in (1,2):
            rows.append({'sequence':seq,'frame_index':i,'num_rois':1 if i==1 else 0,'source_path':f'{seq}_{i}.png'})
    with (proc/'frame_manifest.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    scores={('Test003',1):.2,('Test001',1):.3}
    records,summary=assemble_labeled_frames('Ped1',proc,gt,scores,'skip')
    assert len(records)==1
    assert records[0]['label']==0
    assert summary['unlabeled_sequences_excluded']==['Test001']
    assert summary['frames_with_no_roi']==1
    assert np.isclose(summary['coverage'],.5)
    zero_records,_=assemble_labeled_frames('Ped1',proc,gt,scores,'zero')
    assert len(zero_records)==2 and zero_records[1]['label']==1 and zero_records[1]['score']==0


def test_letterbox_32():
    im=np.full((12,30,3),100,dtype=np.uint8)
    result=letterbox_grayscale(im,32)
    assert result.shape==(32,32)
    assert result.dtype==np.uint8
