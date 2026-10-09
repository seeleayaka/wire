"""One fixed5x5 area feature trial after exact-point pose-sensitivity failure."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from component_cues import gate
from regional_contact import score

ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'artifacts/mendeley_regional_contact_source_20261006';out.mkdir(exist_ok=False)
    paths=[ROOT/'artifacts/mendeley_source_socket_visual_labels_20261005/labels.json',
        ROOT/'artifacts/mendeley_socket_source_phenotype_20261005/protocol.json',
        ROOT/'artifacts/mendeley_positive_contact_source_20261006/fit_contact_support.png',
        ROOT/'artifacts/mendeley_component_rescue_source_20261006/report.json',
        ROOT/'artifacts/mendeley_component_color_source_20261006/report.json']
    byid={r['id']:r for r in json.loads(paths[0].read_text(encoding='utf-8'))['rows']}
    split=json.loads(paths[1].read_text(encoding='utf-8'));support=np.asarray(Image.open(paths[2]))>0
    compound={r['id']:r for r in json.loads(paths[3].read_text(encoding='utf-8'))['cases']}
    baseline=json.loads(paths[4].read_text(encoding='utf-8'))['baseline_LOO_results']
    before=source_pins();pins={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('regional_contact.py'),*paths]}
    offsets=[(x,y) for x in [-2,0,2] for y in [-2,0,2]]
    (out/'protocol.json').write_text(json.dumps(dict(pins=pins,mainline_pins=before,
        pooling='fixed5x5 local bright-neutral area density, no change to image/mask evidence',
        support='prior FIT-only49 support sites retained',
        threshold='max(FIT visible95th percentile,FIT exposed5th percentile)',
        offsets=offsets,source_gate='zero wrong, no old loss, strict gain, all gains stable at every offset',
        no_threshold_or_window_search=True,no_inspection_read=True,originals_fresh_poses_reused=True,
        repeated_development_not_independent_validation=True),indent=2),encoding='utf-8')
    scores={}
    for identity in split['fit_ids']+split['calibration_ids']:
        r=byid[identity]
        if sha256(r['source_path'])!=r['source_sha256']:raise ValueError('source drift')
        rgb=np.asarray(Image.open(r['source_path']).convert('RGB'))
        pose=np.asarray(r['pose']['inspection_to_reference_local']);values=[]
        for dx,dy in offsets:
            matrix=np.array([[1,0,-1600+dx],[0,1,-1000+dy],[0,0,1]])@pose
            patch=cv2.warpPerspective(rgb,matrix,(100,50),flags=cv2.INTER_LINEAR)
            values.append(score(patch,support))
        scores[identity]=values
    center=offsets.index((0,0));fit=split['fit_ids']
    threshold=max(float(np.quantile([scores[i][center] for i in fit if byid[i]['visual_label']==1],.95)),
        float(np.quantile([scores[i][center] for i in fit if byid[i]['visual_label']==0],.05)))
    rows=[];gains=[];losses=[]
    for old in baseline:
        i=old['id'];original=old['prediction']['visual_label_candidate'];proposal=compound[i]['prediction']['visual_label_candidate']
        label=original if original is not None else (0 if proposal==0 and scores[i][center]>threshold else None)
        rows.append(dict(id=i,visual_label=byid[i]['visual_label'],regional_scores=scores[i],
            prediction=dict(visual_label_candidate=label),all_offsets_positive=min(scores[i])>threshold))
        if original is None and label==byid[i]['visual_label']:gains.append(i)
        if original==byid[i]['visual_label'] and label!=original:losses.append(i)
    result=gate(rows);robust=all(min(scores[i])>threshold for i in gains)
    passed=result['passed'] and bool(gains) and not losses and robust
    if before!=source_pins() or any(sha256(p)!=d for p,d in pins.items()):raise ValueError('E/input/code drift')
    report=dict(status='complete',source_gate=result,threshold=threshold,source_gains=gains,source_losses=losses,
        gain_pose_stability_passed=robust,strict_robust_source_gain=passed,cases=rows,
        mainline_unchanged=True,deployed=False,new_confirmed_connections=0)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
if __name__=='__main__':main()
