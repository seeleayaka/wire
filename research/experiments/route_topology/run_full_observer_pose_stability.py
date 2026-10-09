"""Fixed five source controls, nine pose shifts, every appearance head fresh.

No fitting or parameter search. This closes the nominal-head limitation of the
earlier contact-only perturbation audit; all views still constitute one observer.
"""
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from heldout_anchor_pose import localize
from translation_socket_observer import TranslationObserver

ROOT=Path(__file__).resolve().parents[2]

def main():
    out=ROOT/'artifacts/mendeley_full_observer_pose_stability_20261006'
    out.mkdir(exist_ok=False)
    scope_path=ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json'
    control_path=ROOT/'artifacts/mendeley_harness_source_controls_20261006/preparation_report.json'
    scope=json.loads(scope_path.read_text(encoding='utf-8'))
    cases=json.loads(control_path.read_text(encoding='utf-8'))['cases']
    before=source_pins();observer=TranslationObserver()
    pins={**observer.pins,**{str(p):sha256(p) for p in [Path(__file__),scope_path,control_path,
           Path(__file__).with_name('translation_socket_observer.py'),
           Path(__file__).with_name('compound_socket_observer.py')]}}
    offsets=[(x,y) for x in [-2,0,2] for y in [-2,0,2]]
    (out/'protocol.json').write_text(json.dumps(dict(pins=pins,production_pins=before,
        offsets_reference_pixels=offsets,source_controls_only=True,
        no_fitting=True,no_threshold_change=True,no_sam=True,
        fresh_original_pose_and_all_appearance_heads=True,
        multiple_views_are_one_observer=True,deployed=False),indent=2),encoding='utf-8')
    sys.path.insert(0,'E:/PythonProject10/prototype')
    from assembly_auto_review_robust_v3 import automatic_homography
    reference=np.asarray(Image.open(scope['reference_image_path']).convert('RGB'))
    anchor=next(a for a in scope['anchors'] if a['id']=='FAN_CPU')
    rows=[];start=time.monotonic()
    try:
        for case in cases:
            original=case['original_source'];assert sha256(original['path'])==original['image_sha256']
            pins[original['path']]=original['image_sha256']
            rgb=np.asarray(Image.open(original['path']).convert('RGB'))
            expected=1 if case['source_control_annotation'] is None else case['source_control_annotation']['visual_label']
            if case['id']=='reference':matrix=np.eye(3)
            else:
                _,registration=automatic_homography(reference[:,:,::-1].copy(),rgb[:,:,::-1].copy())
                assert registration['alignment_quality']['reliable']
                pose=localize(reference,rgb,anchor,np.asarray(registration['source_to_reference_homography']))
                assert pose['localization_proposal_supported']
                matrix=np.asarray(pose['inspection_to_reference_local'])
            predictions=[]
            for x,y in offsets:
                shifted=np.array([[1,0,x],[0,1,y],[0,0,1]])@matrix
                prediction,patch=observer.infer_original(rgb,shifted)
                predictions.append(dict(offset=[x,y],prediction=prediction))
                Image.fromarray(patch).save(out/f"{case['id']}_x{x}_y{y}.png")
                (out/'progress.json').write_text(json.dumps(dict(status='running',case=case['id'],
                    completed_sources=len(rows),completed_views=len(predictions),seconds=time.monotonic()-start)),encoding='utf-8')
            labels=[p['prediction']['visual_label_candidate'] for p in predictions]
            rows.append(dict(id=case['id'],source=original,expected_source_visual_label=expected,
                fresh_nominal_pose=matrix.tolist(),predictions=predictions,
                nominal_correct=labels[4]==expected,
                wrong_offsets=[p['offset'] for p in predictions if p['prediction']['visual_label_candidate'] not in [None,expected]],
                abstained_offsets=[p['offset'] for p in predictions if p['prediction']['visual_label_candidate'] is None],
                all_offsets_correct=all(label==expected for label in labels)))
        assert source_pins()==before
        assert all(sha256(p)==d for p,d in pins.items())
        report=dict(status='complete',cases=rows,fresh_complete_head_inferences=len(rows)*9,
            all_fixed_source_offsets_correct=all(r['all_offsets_correct'] for r in rows),
            wrong_offset_count=sum(len(r['wrong_offsets']) for r in rows),
            abstained_offset_count=sum(len(r['abstained_offsets']) for r in rows),
            source_control_reliability_probe_not_blind_accuracy=True,
            production_unchanged=True,seconds=time.monotonic()-start,deployed=False)
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (out/'progress.json').write_text(json.dumps(dict(status='complete')),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)
    except BaseException as error:
        (out/'progress.json').write_text(json.dumps(dict(status='failed',error=str(error))),encoding='utf-8')
        (out/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise

if __name__=='__main__':main()
