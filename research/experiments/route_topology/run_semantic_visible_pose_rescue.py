"""One asymmetric candidate experiment: only stable visible-body semantic rescue.

Preserve every prior candidate; never rescue exposed class with this route.
Every CAL image and all9 poses freshly encoded, self excluded from conformal ranks.
No threshold fitting, no new independent votes, no topology/occupancy assertion.
"""
import json
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from compound_socket_observer import Observer
from component_cues import prediction, gate

ROOT=Path(__file__).resolve().parents[2]

def main():
    out=ROOT/'artifacts/mendeley_semantic_visible_pose_rescue_20261006';out.mkdir(exist_ok=False)
    lp=ROOT/'artifacts/mendeley_source_socket_visual_labels_20261005/labels.json'
    bp=ROOT/'artifacts/mendeley_contact_translation_nuisance_20261006/report.json'
    rows_byid={r['id']:r for r in json.loads(lp.read_text(encoding='utf-8'))['rows']}
    baseline=json.loads(bp.read_text(encoding='utf-8'))['cases']
    before=source_pins();observer=Observer()
    raw=observer.raw_semantic
    model={k:np.asarray(raw[k]) if k in ['center','scale','weights'] else raw[k] for k in ['center','scale','weights','bias']}
    calibration={int(k):v for k,v in raw['calibration'].items()}
    offsets=[(x,y) for x in [-2,0,2] for y in [-2,0,2]]
    pins={**observer.pins,**{str(p):sha256(p) for p in [Path(__file__),lp,bp,
        Path(__file__).with_name('compound_socket_observer.py'),Path(__file__).with_name('component_cues.py')]}}
    (out/'protocol.json').write_text(json.dumps(dict(pins=pins,production_pins=before,
        fixed_offsets=offsets,frozen_semantic_model=True,leave_self_out=True,
        proposal='old unknown and all nine semantic singleton1 => visible-body candidate1 only',
        preserve_old=True,never_rescue_class0=True,class_tail_alpha=.05,
        no_search_or_threshold_change=True,one_observer=True,
        all80_source_CAL_originals_and_DINO_fresh_existing_poses_reused=True,
        no_demo_or_field_validation=True,source_development_reused=True,
        zero_wrong_strict_gain_required=True,not_occupancy_or_topology=True,deployed=False),indent=2),encoding='utf-8')
    # Observer.infer supplies the exact same frozen DINO descriptor; retain tokens'
    # derived probabilities and recompute LOO ranks independently of its nominal label.
    # The descriptor is extracted here directly so calibration exclusion is explicit.
    from component_cues import semantic_descriptor
    import torch
    results=[];gains=[];start=time.monotonic()
    for old in baseline:
        i=old['id'];r=rows_byid[i];assert sha256(r['source_path'])==r['source_sha256']
        pins[r['source_path']]=r['source_sha256']
        image=np.asarray(Image.open(r['source_path']).convert('RGB'))
        pose=np.asarray(r['pose']['inspection_to_reference_local']);valid_source=np.ones(image.shape[:2],np.uint8)
        probes=[]
        for x,y in offsets:
            h=np.array([[1,0,-1600+x],[0,1,-1000+y],[0,0,1]])@pose
            assert cv2.warpPerspective(valid_source,h,(100,50),flags=cv2.INTER_NEAREST).all()
            patch=cv2.warpPerspective(image,h,(100,50),flags=cv2.INTER_LINEAR)
            resized=np.asarray(Image.fromarray(patch).resize((224,112),Image.Resampling.BILINEAR)).copy()
            tensor=(torch.from_numpy(resized.transpose(2,0,1)).float()/255-observer.mean)/observer.std
            with torch.inference_mode():tokens=observer.encoder.forward_features(tensor.unsqueeze(0))['x_norm_patchtokens'].squeeze(0).cpu().numpy()
            tokens=tokens/np.maximum(np.linalg.norm(tokens,axis=1,keepdims=True),1e-8)
            feature,_=semantic_descriptor(tokens,np.asarray(raw['centers']))
            probes.append(dict(offset=[x,y],prediction=prediction(model,feature,calibration,excluded_id=i)))
        previous=old['prediction']['visual_label_candidate']
        positive=all(p['prediction']['visual_label_candidate']==1 for p in probes)
        final=previous if previous is not None else (1 if positive else None)
        results.append(dict(id=i,visual_label=r['visual_label'],old_candidate=previous,
            semantic_probes=probes,stable_visible_body_proposal=positive,
            prediction=dict(visual_label_candidate=final)))
        if previous is None and final==r['visual_label']:gains.append(i)
        (out/'progress.json').write_text(json.dumps(dict(status='fresh_DINO_source_probes',completed=len(results),total=80)),encoding='utf-8')
    assert source_pins()==before and all(sha256(p)==d for p,d in pins.items())
    score=gate(results)
    report=dict(status='complete',source_gate=score,gains=gains,old_losses=[],
        strict_net_source_gain=score['passed'] and bool(gains),cases=results,
        fresh_DINO_encodings=720,production_unchanged=True,
        seconds=time.monotonic()-start,deployed=False)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'progress.json').write_text(json.dumps(dict(status='complete')),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)

if __name__=='__main__':main()
