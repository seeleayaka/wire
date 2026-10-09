"""FIT-only pose-averaged LAB/edge heads; frozen-source net-gain gate first.

Fixed +/-2px feature averaging, no image IDs/labels as inference features.
All prior candidates preserved. A new label requires three heads and nine outer
offsets to agree; they remain one observer, not votes. Reused CAL is development.
"""
import json
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from socket_appearance import descriptor
from socket_phenotype_agreement import feature_view, agreement
from component_cues import fit_head, probability, prediction, gate

ROOT=Path(__file__).resolve().parents[2]

def main():
    out=ROOT/'artifacts/mendeley_mean_pose_source_20261006';out.mkdir(exist_ok=False)
    lp=ROOT/'artifacts/mendeley_source_socket_visual_labels_20261005/labels.json'
    sp=ROOT/'artifacts/mendeley_socket_source_phenotype_20261005/protocol.json'
    bp=ROOT/'artifacts/mendeley_contact_translation_nuisance_20261006/report.json'
    byid={r['id']:r for r in json.loads(lp.read_text(encoding='utf-8'))['rows']}
    split=json.loads(sp.read_text(encoding='utf-8'));baseline=json.loads(bp.read_text(encoding='utf-8'))['cases']
    before=source_pins();pins={str(p):sha256(p) for p in [Path(__file__),lp,sp,bp,
        Path(__file__).with_name('socket_appearance.py'),Path(__file__).with_name('component_cues.py'),
        Path(__file__).with_name('socket_phenotype_agreement.py')]}
    offsets=[(x,y) for x in [-2,0,2] for y in [-2,0,2]]
    grid=[(x,y) for x in [-4,-2,0,2,4] for y in [-4,-2,0,2,4]]
    (out/'protocol.json').write_text(json.dumps(dict(pins=pins,production_pins=before,
        fit_only_ids=split['fit_ids'],calibration_ids=split['calibration_ids'],
        feature='mean of nine320D native-crop descriptors at fixed +/-2 reference pixels',
        source_originals_fresh_existing_poses_reused=True,
        calibration_self_excluded=True,class_tail_alpha=.05,
        all_three_heads_and_all_nine_outer_offsets_same_required_for_rescue=True,
        preserve_all_old_candidates=True,strict_net_gain_zero_wrong_required=True,
        no_parameter_search=True,no_demo_images=True,no_sam=True,
        one_observer=True,not_topology_or_field_accuracy=True,deployed=False),indent=2),encoding='utf-8')
    start=time.monotonic();features={};cv2.setNumThreads(1)
    for i in split['fit_ids']+split['calibration_ids']:
        r=byid[i];assert sha256(r['source_path'])==r['source_sha256']
        pins[r['source_path']]=r['source_sha256']
        rgb=np.asarray(Image.open(r['source_path']).convert('RGB'))
        pose=np.asarray(r['pose']['inspection_to_reference_local'])
        values={}
        valid_input=np.ones(rgb.shape[:2],np.uint8)
        for x,y in grid:
            h=np.array([[1,0,-1600+x],[0,1,-1000+y],[0,0,1]])@pose
            valid=cv2.warpPerspective(valid_input,h,(100,50),flags=cv2.INTER_NEAREST)
            if not valid.all():raise ValueError('missing native source pixels')
            patch=cv2.warpPerspective(rgb,h,(100,50),flags=cv2.INTER_LINEAR)
            values[x,y]=descriptor(patch)
        features[i]=np.asarray([np.mean([values[x+dx,y+dy] for dx,dy in offsets],axis=0) for x,y in offsets])
        (out/'progress.json').write_text(json.dumps(dict(status='source_features',completed=len(features),total=197)),encoding='utf-8')
    heads={};calibrations={};fit=split['fit_ids'];cal=split['calibration_ids']
    for view in ['combined','color','edge']:
        heads[view]=fit_head([feature_view(features[i][4],view) for i in fit],[byid[i]['visual_label'] for i in fit])
        calibration={0:[],1:[]}
        for i in cal:
            k=byid[i]['visual_label'];p=probability(heads[view],feature_view(features[i][4],view))
            calibration[k].append(dict(id=i,score=p if k==0 else 1-p))
        calibrations[view]=calibration
    rows=[];gains=[];raw=[]
    for old in baseline:
        i=old['id'];label=byid[i]['visual_label'];probes=[]
        for j,offset in enumerate(offsets):
            heads_result={v:prediction(heads[v],feature_view(features[i][j],v),calibrations[v],excluded_id=i) for v in heads}
            probes.append(dict(offset=offset,**agreement(heads_result)))
        candidates=[p['visual_label_candidate'] for p in probes]
        proposed=candidates[4] if candidates[4] is not None and all(c==candidates[4] for c in candidates) else None
        previous=old['prediction']['visual_label_candidate'];accepted=previous if previous is not None else proposed
        raw.append(dict(id=i,visual_label=label,prediction=dict(visual_label_candidate=proposed)))
        rows.append(dict(id=i,visual_label=label,old_candidate=previous,proposed=proposed,
            prediction=dict(visual_label_candidate=accepted),outer_probes=probes))
        if previous is None and accepted==label:gains.append(i)
    result=gate(rows)
    assert source_pins()==before and all(sha256(p)==d for p,d in pins.items())
    model={v:{**{k:val.tolist() if isinstance(val,np.ndarray) else val for k,val in heads[v].items()},
        'calibration':calibrations[v]} for v in heads}
    (out/'model.json').write_text(json.dumps(model,indent=2),encoding='utf-8')
    report=dict(status='complete',source_gate=result,raw_new_head_gate=gate(raw),gains=gains,
        old_losses=[],strict_net_source_gain=result['passed'] and bool(gains),cases=rows,
        production_unchanged=True,seconds=time.monotonic()-start,deployed=False)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'progress.json').write_text(json.dumps(dict(status='complete')),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)

if __name__=='__main__':main()
