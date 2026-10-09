"""Fresh original and global/local pose checks for the frozen contact cue."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from heldout_anchor_pose import localize
from run_positive_contact_gate import bright_contacts

ROOT = Path(__file__).resolve().parents[2]

def main():
    out = ROOT/'artifacts/mendeley_positive_contact_fresh_controls_20261006'
    out.mkdir(exist_ok=False)
    before = source_pins()
    scope_path = ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json'
    prep_path = ROOT/'artifacts/mendeley_harness_source_controls_20261006/preparation_report.json'
    report_path = ROOT/'artifacts/mendeley_positive_contact_source_20261006/report.json'
    mask_path = report_path.parent/'fit_contact_support.png'
    pins = {str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('run_positive_contact_gate.py'),scope_path,prep_path,report_path,mask_path]}
    source_report=json.loads(report_path.read_text(encoding='utf-8'))
    if not source_report['strict_net_source_gain']: raise ValueError('source gate failed')
    scope=json.loads(scope_path.read_text(encoding='utf-8'))
    sources=json.loads(prep_path.read_text(encoding='utf-8'))['cases']
    support=np.asarray(Image.open(mask_path))>0
    threshold=source_report['fit_threshold']
    (out/'protocol.json').write_text(json.dumps(dict(pins=pins,mainline_pins=before,
        rule='fixed FIT support and threshold; fresh originals/global and local CPU pose',
        no_threshold_changes=True,cue_only_not_full_classifier_or_topology=True),indent=2),encoding='utf-8')
    sys.path.insert(0,'E:/PythonProject10/prototype')
    from assembly_auto_review_robust_v3 import automatic_homography
    reference=np.asarray(Image.open(scope['reference_image_path']).convert('RGB'))
    anchor=next(a for a in scope['anchors'] if a['id']=='FAN_CPU')
    rows=[]
    for r in sources:
        origin=r['original_source']
        if sha256(origin['path'])!=origin['image_sha256']:raise ValueError('original drift')
        rgb=np.asarray(Image.open(origin['path']).convert('RGB'))
        if r['id']=='reference':pose=dict(localization_proposal_supported=True,inspection_to_reference_local=np.eye(3).tolist())
        else:
            _,reg=automatic_homography(reference[:,:,::-1].copy(),rgb[:,:,::-1].copy())
            if not reg['alignment_quality']['reliable']:raise ValueError('global pose unsupported')
            pose=localize(reference,rgb,anchor,np.array(reg['source_to_reference_homography']))
        if not pose['localization_proposal_supported']:raise ValueError('local pose unsupported')
        matrix=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@np.array(pose['inspection_to_reference_local'])
        valid=cv2.warpPerspective(np.ones(rgb.shape[:2],np.uint8),matrix,(100,50),flags=cv2.INTER_NEAREST)
        if not valid.all():raise ValueError('missing pixels')
        patch=cv2.warpPerspective(rgb,matrix,(100,50),flags=cv2.INTER_LINEAR)
        Image.fromarray(patch).save(out/(r['id']+'_fresh_socket.png'))
        score=float(bright_contacts(patch)[support].mean())
        annotation=r['source_control_annotation']
        label=1 if annotation is None else annotation['visual_label']
        rows.append(dict(id=r['id'],source=origin,pose=pose,visual_label=label,
            contact_score=score,positive_contact_cue=score>threshold))
    if before!=source_pins() or any(sha256(p)!=d for p,d in pins.items()):raise ValueError('E/code drift')
    false_cues=[r['id'] for r in rows if r['visual_label']==1 and r['positive_contact_cue']]
    missed=[r['id'] for r in rows if r['visual_label']==0 and not r['positive_contact_cue']]
    report=dict(status='complete',cases=rows,visible_false_cues=false_cues,exposed_missed=missed,
        fixed_source_cue_gate=not false_cues and not missed,mainline_unchanged=True,deployed=False,
        new_confirmed_connections=0,not_full_classifier_acceptance=True)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__': main()
