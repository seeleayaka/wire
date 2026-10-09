"""Fresh fixed5 originals: localization reliability, never topology/field accuracy."""
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from core import sha256
from heldout_anchor_pose import localize
from root_static_pose import localize_root
from local_anchor_pose import project_points
from run_audit import source_pins
from run_review import save

ROOT=Path(__file__).resolve().parents[2]


def main():
    out=ROOT/'artifacts/mendeley_root_pose_controls_20261007'; out.mkdir(exist_ok=False)
    sp=ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json'
    cp=ROOT/'artifacts/mendeley_harness_source_controls_20261006/preparation_report.json'
    scope=json.loads(sp.read_text(encoding='utf-8'))
    rows=json.loads(cp.read_text(encoding='utf-8'))['cases']
    before=source_pins(); pins={str(p):sha256(p) for p in [sp,cp,Path(__file__),
        Path(__file__).with_name('root_static_pose.py'),Path(__file__).with_name('heldout_anchor_pose.py')]}
    assert len(rows)==5
    save(out/'protocol.json',dict(pins=pins,mainline_pins=before,
        source_ids=[r['id'] for r in rows],source_only=True,
        one_variable='L1_square_root_descriptor_normalization',
        changes_to_matching_geometric_heldout_thresholds=False,
        views=['original','brightness90','brightness110'],perturbation_predeclared=True,
        synthetic_photometric_views_not_independent_observers=True,
        pose_reference='fresh nominal baseline pose, not independently certified GT',
        stable_definition='all nominal/brightness poses supported and max anchor corner drift<=2.5px',
        acceptance='strict increase in stable source anchors; zero loss of old stable anchors; all five retained',
        no_demo_before_source_gain=True,no_SAM_or_training=True,deployed=False))
    reference=np.asarray(Image.open(scope['reference_image_path']).convert('RGB'))
    sys.path.insert(0,'E:/PythonProject10/prototype')
    from assembly_auto_review_robust_v3 import automatic_homography
    results=[]; start=time.monotonic()
    for row in rows:
        origin=row['original_source']; assert sha256(origin['path'])==origin['image_sha256']
        rgb=np.asarray(Image.open(origin['path']).convert('RGB'))
        if row['id']=='reference': global_matrix=np.eye(3)
        else:
            _,registration=automatic_homography(reference[:,:,::-1].copy(),rgb[:,:,::-1].copy())
            assert registration['alignment_quality']['reliable']
            global_matrix=np.asarray(registration['source_to_reference_homography'])
        views={'original':rgb,'brightness90':np.clip(rgb.astype(float)*.9,0,255).astype(np.uint8),
               'brightness110':np.clip(rgb.astype(float)*1.1,0,255).astype(np.uint8)}
        anchors=[]
        for anchor in scope['anchors']:
            probes=[]
            for name,view in views.items():
                cv2.setRNGSeed(0); old=localize(reference,view,anchor,global_matrix)
                cv2.setRNGSeed(0); new=localize_root(reference,view,anchor,global_matrix)
                probes.append(dict(view=name,old=old,candidate=new))
            baseline=probes[0]['old']; l,t,r,b=anchor['bbox_xyxy']; corners=[[l,t],[r,t],[r,b],[l,b]]
            target=project_points(corners,np.linalg.inv(baseline['inspection_to_reference_local'])) if baseline['localization_proposal_supported'] else None
            stable={}; max_drift={}
            for method in ['old','candidate']:
                drifts=[]
                for probe in probes:
                    pose=probe[method]
                    if target is None or not pose['localization_proposal_supported']:
                        drifts.append(None)
                    else:
                        actual=project_points(corners,np.linalg.inv(pose['inspection_to_reference_local']))
                        drifts.append(float(np.linalg.norm(actual-target,axis=1).max()))
                stable[method]=all(d is not None and d<=2.5 for d in drifts)
                max_drift[method]=max(drifts) if all(d is not None for d in drifts) else None
            anchors.append(dict(id=anchor['id'],probes=probes,stable=stable,max_corner_drift_px=max_drift))
        results.append(dict(id=row['id'],original_source=origin,global_matrix=global_matrix.tolist(),anchors=anchors))
        save(out/'progress.json',dict(status='running',completed=len(results),total=5))
    old_count=sum(a['stable']['old'] for r in results for a in r['anchors'])
    new_count=sum(a['stable']['candidate'] for r in results for a in r['anchors'])
    losses=[dict(case=r['id'],anchor=a['id']) for r in results for a in r['anchors'] if a['stable']['old'] and not a['stable']['candidate']]
    gains=[dict(case=r['id'],anchor=a['id']) for r in results for a in r['anchors'] if not a['stable']['old'] and a['stable']['candidate']]
    assert source_pins()==before and all(sha256(p)==d for p,d in pins.items())
    passed=bool(gains) and not losses
    save(out/'report.json',dict(status='complete',source_reliability_gate_passed=passed,
        old_stable_anchors=old_count,candidate_stable_anchors=new_count,source_gains=gains,source_losses=losses,
        cases=results,seconds=time.monotonic()-start,originals_decoded_fresh=5,
        local_feature_runs=60,fresh_global_registration=True,global_seed_reused_within_photometric_views=True,
        geometric_localization_not_identity_or_topology=True,new_topology_hits=0,deployed=False))
    save(out/'progress.json',dict(status='complete',source_reliability_gate_passed=passed))
    print(json.dumps(dict(passed=passed,old_stable=old_count,new_stable=new_count,gains=gains,losses=losses,
                         seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__': main()
