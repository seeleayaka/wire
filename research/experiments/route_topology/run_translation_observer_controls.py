"""Fresh source controls and30-photo development preview for robust observer."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from translation_socket_observer import TranslationObserver
from core import sha256
from run_audit import source_pins
from heldout_anchor_pose import localize

ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'artifacts/mendeley_translation_observer_fresh_20261006';out.mkdir(exist_ok=False)
    scope_path=ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json'
    control_path=ROOT/'artifacts/mendeley_harness_source_controls_20261006/preparation_report.json'
    demo_path=ROOT/'artifacts/mendeley_compound_fresh30_20261006/report.json'
    scope=json.loads(scope_path.read_text(encoding='utf-8'))
    sources=[dict(id=r['id'],dataset='source_controls',origin=r['original_source'],label=1 if r['source_control_annotation'] is None else r['source_control_annotation']['visual_label']) for r in json.loads(control_path.read_text(encoding='utf-8'))['cases']]
    sources.extend(dict(id=r['id'],dataset='demo30',origin=r['source'],label=None) for r in json.loads(demo_path.read_text(encoding='utf-8'))['cases'] if r['id']!='reference')
    before=source_pins();observer=TranslationObserver()
    pins={**observer.pins,**{str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('translation_socket_observer.py'),Path(__file__).with_name('compound_socket_observer.py'),scope_path,control_path,demo_path]}}
    pins.update({r['origin']['path']:r['origin']['image_sha256'] for r in sources})
    (out/'protocol.json').write_text(json.dumps(dict(pins=pins,mainline_pins=before,
        original_decode_pose_encoder_fresh=True,source_controls_and_development_demo_not_blind_test=True,
        no_test_GT_read=True,threshold_frozen=True,no_sam_inference=True),indent=2),encoding='utf-8')
    sys.path.insert(0,'E:/PythonProject10/prototype')
    from assembly_auto_review_robust_v3 import automatic_homography
    reference=np.asarray(Image.open(scope['reference_image_path']).convert('RGB'))
    anchor=next(a for a in scope['anchors'] if a['id']=='FAN_CPU');rows=[]
    for r in sources:
        rgb=np.asarray(Image.open(r['origin']['path']).convert('RGB'))
        if r['id']=='reference':pose=dict(inspection_to_reference_local=np.eye(3).tolist(),localization_proposal_supported=True)
        else:
            _,reg=automatic_homography(reference[:,:,::-1].copy(),rgb[:,:,::-1].copy())
            if not reg['alignment_quality']['reliable']:raise ValueError('global pose unsupported')
            pose=localize(reference,rgb,anchor,np.asarray(reg['source_to_reference_homography']))
        if not pose['localization_proposal_supported']:raise ValueError('local pose unsupported')
        prediction,patch=observer.infer_original(rgb,pose['inspection_to_reference_local'])
        Image.fromarray(patch).save(out/(r['dataset']+'_'+r['id']+'_socket.png'))
        rows.append(dict(**r,pose=pose,prediction=prediction))
        (out/'progress.json').write_text(json.dumps(dict(status='running',completed=len(rows),total=len(sources))),encoding='utf-8')
    controls=[r for r in rows if r['dataset']=='source_controls'];demo=[r for r in rows if r['dataset']=='demo30']
    if before!=source_pins() or any(sha256(p)!=d for p,d in pins.items()):raise ValueError('source/code/E drift')
    report=dict(status='complete',cases=rows,source_controls_passed=all(r['prediction']['visual_label_candidate']==r['label'] for r in controls),
        demo_counts={str(k):sum(r['prediction']['visual_label_candidate']==k for r in demo) for k in [0,1,None]},
        new_candidates=[r['id'] for r in demo if r['prediction']['old_candidate'] is None and r['prediction']['visual_label_candidate'] is not None],
        prior_supported_loss=[r['id'] for r in demo if r['prediction']['old_candidate'] is not None and r['prediction']['visual_label_candidate']!=r['prediction']['old_candidate']],
        mainline_unchanged=True,deployed=False,electrical_or_topology_acceptance=False)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'progress.json').write_text(json.dumps(dict(status='complete',completed=len(rows))),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
if __name__=='__main__':main()
