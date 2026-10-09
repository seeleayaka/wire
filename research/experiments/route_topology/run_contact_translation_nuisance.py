"""FIT-calibrated bounded pose-nuisance marginalization, no image-name features."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from component_cues import gate
from run_positive_contact_gate import bright_contacts

ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'artifacts/mendeley_contact_translation_nuisance_20261006';out.mkdir(exist_ok=False)
    lp=ROOT/'artifacts/mendeley_source_socket_visual_labels_20261005/labels.json'
    sp=ROOT/'artifacts/mendeley_socket_source_phenotype_20261005/protocol.json'
    mp=ROOT/'artifacts/mendeley_positive_contact_source_20261006/fit_contact_support.png'
    cp=ROOT/'artifacts/mendeley_component_rescue_source_20261006/report.json'
    bp=ROOT/'artifacts/mendeley_component_color_source_20261006/report.json'
    byid={r['id']:r for r in json.loads(lp.read_text(encoding='utf-8'))['rows']}
    split=json.loads(sp.read_text(encoding='utf-8'));support=np.asarray(Image.open(mp))>0
    proposals={r['id']:r for r in json.loads(cp.read_text(encoding='utf-8'))['cases']}
    old=json.loads(bp.read_text(encoding='utf-8'))['baseline_LOO_results']
    before=source_pins();pins={str(p):sha256(p) for p in [Path(__file__),lp,sp,mp,cp,bp]}
    offsets=[(x,y) for x in [-2,0,2] for y in [-2,0,2]]
    grid=[(x,y) for x in [-4,-2,0,2,4] for y in [-4,-2,0,2,4]]
    (out/'protocol.json').write_text(json.dumps(dict(pins=pins,mainline_pins=before,
        fit_feature='max contact-support fraction within fixed3x3 translation nuisance grid +/-2px',
        fit_threshold='max(FIT visible95th percentile,FIT exposed5th percentile), not inherited original threshold',
        robustness='repeat same max feature for each outer +/-2px diagnostic displacement, using fixed5x5 combined grid',
        zero_wrong_no_old_loss_strict_gain_required=True,all_gains_robust_required=True,
        no_parameter_search=True,no_new_independent_observers=True,no_new_connectivity_pixels=True,
        all197_originals_fresh_existing_poses_reused=True,development_CAL_reused=True),indent=2),encoding='utf-8')
    features={}
    for i in split['fit_ids']+split['calibration_ids']:
        r=byid[i]
        if sha256(r['source_path'])!=r['source_sha256']:raise ValueError('source drift')
        rgb=np.asarray(Image.open(r['source_path']).convert('RGB'));pose=np.asarray(r['pose']['inspection_to_reference_local'])
        values={}
        for x,y in grid:
            matrix=np.array([[1,0,-1600+x],[0,1,-1000+y],[0,0,1]])@pose
            patch=cv2.warpPerspective(rgb,matrix,(100,50),flags=cv2.INTER_LINEAR)
            values[(x,y)]=float(bright_contacts(patch)[support].mean())
        features[i]=[max(values[(x+dx,y+dy)] for dx,dy in offsets) for x,y in offsets]
    center=offsets.index((0,0));fit=split['fit_ids']
    threshold=max(float(np.quantile([features[i][center] for i in fit if byid[i]['visual_label']==1],.95)),
        float(np.quantile([features[i][center] for i in fit if byid[i]['visual_label']==0],.05)))
    rows=[];gains=[];losses=[]
    for r in old:
        i=r['id'];label=byid[i]['visual_label'];previous=r['prediction']['visual_label_candidate']
        proposed=proposals[i]['prediction']['visual_label_candidate']
        accepted=previous if previous is not None else (0 if proposed==0 and features[i][center]>threshold else None)
        rows.append(dict(id=i,visual_label=label,nuisance_scores=features[i],
            prediction=dict(visual_label_candidate=accepted),all_outer_offsets_positive=min(features[i])>threshold))
        if previous is None and accepted==label:gains.append(i)
        if previous==label and accepted!=previous:losses.append(i)
    result=gate(rows);stable=all(min(features[i])>threshold for i in gains)
    if before!=source_pins() or any(sha256(p)!=d for p,d in pins.items()):raise ValueError('E/code/input drift')
    report=dict(status='complete',source_gate=result,threshold=threshold,gains=gains,losses=losses,
        all_gains_pose_stable=stable,strict_robust_source_gain=result['passed'] and bool(gains) and not losses and stable,
        cases=rows,mainline_unchanged=True,deployed=False,new_confirmed_connections=0)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
if __name__=='__main__':main()
