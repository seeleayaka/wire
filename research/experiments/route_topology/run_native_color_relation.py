"""Fresh five-original native-colour continuity readiness; no topology verdict.

Unwarped source pixels, no morphology/closing/curve fit. A continuous union of
red/yellow/blue is not proof of individual-wire identity at crossings.
"""
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image
from core import sha256
from component_cues import color_components
from heldout_anchor_pose import localize
from prepare_mendeley_scope import inspection_scope
from run_audit import source_pins

ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'artifacts/mendeley_native_color_relation_20261006';out.mkdir(exist_ok=False)
    start=time.monotonic();before=source_pins()
    scope_path=ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json'
    scope=json.loads(scope_path.read_text(encoding='utf-8'))
    approval=json.loads((scope_path.parent/'approval_record.json').read_text(encoding='utf-8'))
    if before!=approval['mainline_pins']:raise ValueError('E drift')
    prep_path=ROOT/'artifacts/mendeley_harness_source_controls_20261006/preparation_report.json'
    sources=json.loads(prep_path.read_text(encoding='utf-8'))['cases']
    pins={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('component_cues.py'),scope_path,prep_path]}
    pins.update({r['original_source']['path']:r['original_source']['image_sha256'] for r in sources})
    (out/'protocol.json').write_text(json.dumps(dict(pins=pins,mainline_pins=before,
        colors=['red','yellow','blue'],HSV_saturation_min=80,HSV_value_min=40,
        no_gap_closing=True,original_pixels_not_warped=True,fresh_pose=True,
        gate='all visible-source controls preserve a continuous component spanning both anchors',
        components_are_colour_unions_not_individual_wire_instances=True),indent=2),encoding='utf-8')
    sys.path.insert(0,'E:/PythonProject10/prototype')
    from assembly_auto_review_robust_v3 import automatic_homography
    reference=np.asarray(Image.open(scope['reference_image_path']).convert('RGB'))
    rows=[]
    for source in sources:
        identity=source['id'];rgb=np.asarray(Image.open(source['original_source']['path']).convert('RGB'))
        if identity=='reference':
            crop_box=[1380,870,1780,1270]
            poses={a['id']:dict(localization_proposal_supported=True,inspection_to_reference_local=np.eye(3).tolist()) for a in scope['anchors']}
        else:
            _,reg=automatic_homography(reference[:,:,::-1].copy(),rgb[:,:,::-1].copy())
            if not reg['alignment_quality']['reliable']:raise ValueError('source global registration failed')
            h=np.array(reg['source_to_reference_homography']);crop_box=inspection_scope([1380,870,1780,1270],h,[rgb.shape[1],rgb.shape[0]])
            poses={a['id']:localize(reference,rgb,a,h) for a in scope['anchors']}
        l,t,r,b=crop_box;crop=rgb[t:b,l:r].copy();categories=color_components(crop)
        active=np.isin(categories,[0,1,3]);Image.fromarray(active.astype(np.uint8)*255).save(out/(identity+'_native_colour_union.png'))
        overlay=crop.copy();overlay[active]=(crop[active]*.4+np.array([255,0,255])*.6).astype(np.uint8)
        Image.fromarray(overlay).save(out/(identity+'_overlay.png'))
        n,labels=cv2.connectedComponents(active.astype(np.uint8),connectivity=8);components=[]
        for k in range(1,n):
            ys,xs=np.nonzero(labels==k);points=np.stack([xs+l,ys+t,np.ones(len(xs))],axis=1);hits={}
            for anchor in scope['anchors']:
                pose=poses[anchor['id']];hits[anchor['id']]=0
                if not pose['localization_proposal_supported']:continue
                mapped=(np.array(pose['inspection_to_reference_local'])@points.T).T;xy=mapped[:,:2]/mapped[:,2:]
                al,at,ar,ab=anchor['bbox_xyxy'];hits[anchor['id']]=int(((xy[:,0]>=al)&(xy[:,0]<=ar)&(xy[:,1]>=at)&(xy[:,1]<=ab)).sum())
            components.append(dict(pixels=len(xs),anchor_hits=hits,spans_two_anchors=all(hits.values())))
        rows.append(dict(id=identity,original=source['original_source'],crop_box=crop_box,fresh_poses=poses,
            components=components,two_anchor_components=sum(c['spans_two_anchors'] for c in components),
            source_annotation=source['source_control_annotation'],decision='insufficient_evidence'))
    visible=[row for row in rows if row['source_annotation'] is None or row['source_annotation']['visual_label']==1]
    passed=all(row['two_anchor_components']==1 for row in visible)
    if before!=source_pins() or any(sha256(p)!=d for p,d in pins.items()):raise ValueError('source/code/E drift')
    report=dict(status='complete',cases=rows,visible_source_continuity_gate=passed,seconds=time.monotonic()-start,
        added_gap_pixels=0,new_confirmed_connections=0,deployed=False,mainline_unchanged=True,
        not_electrical_or_single_wire_identity=True,inspection_images_or_labels_read=False,
        source_labels_not_human_GT=True,next='independent native pixel audit needed' if passed else 'reject before inspection; no gap filling')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='complete',gate=passed,seconds=report['seconds'],spanning_components={r['id']:r['two_anchor_components'] for r in rows})))
if __name__=='__main__':main()
