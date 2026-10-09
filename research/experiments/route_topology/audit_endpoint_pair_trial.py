"""Independent native flood-fill + scalar endpoint-hit replay of all5 sources."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from audit_semantic_visible_bundle_native import flood_components
from audit_cable_socket_controls import replay_hits
from run_review import verified_run
from run_prompt_contrast import save,digest,verify
from bundle_runtime_pins import source_pins

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
OUT=ROOT/'artifacts/endpoint_pair_trial_v3_20261008'


def main():
    protocol=json.loads((OLD/'protocol.json').read_text(encoding='utf-8'))
    verify(protocol['pins'])
    scope=json.loads(Path(protocol['confirmed_scope_path']).read_text(encoding='utf-8'))
    prepared=json.loads((OLD/'preparation_report.json').read_text(encoding='utf-8'))['cases']
    report=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
    assert report['status']=='complete' and len(report['cases'])==5
    checks=[]
    for case in prepared:
        row=next(r for r in report['cases'] if r['id']==case['id'])
        context=case['crop_context'];crop=context['crop_box_xyxy']
        assert digest(case['original_source']['path'])==case['original_source']['image_sha256']
        pixels=np.asarray(Image.open(case['original_source']['path']).convert('RGB').crop(crop))
        assert np.array_equal(pixels,np.asarray(Image.open(context['source']['path']).convert('RGB')))
        inventory=verified_run(OLD/case['id']/'cable_plus_reference_anatomy_box',context['source']['path'])
        candidates=[]
        for path,score in zip(inventory['paths'],inventory['scores']):
            raw=np.asarray(Image.open(path).convert('L'))>0
            ys,xs=np.where(raw);h,w=raw.shape
            boundary=bool(len(xs) and (xs.min()<=1 or ys.min()<=1 or xs.max()>=w-2 or ys.max()>=h-2))
            signatures=replay_hits(flood_components(raw),w,crop[:2],scope['anchors'],case['anchors'])
            coverage={a['id']:[i for i,(_,hits) in enumerate(signatures,1) if hits[a['id']]>0] for a in scope['anchors']}
            if score>=.75 and not boundary and all(coverage.values()):
                candidates.append(dict(record_id=path.stem,anchor_components=coverage))
        if case['phenotype']=='mating_body_visible':
            expected='reference_endpoint_pair_candidate' if len(candidates)==1 else 'insufficient_endpoint_evidence'
            assert expected==row['decision']
            assert [{k:r[k] for k in ['record_id','anchor_components']} for r in row['candidate_records']]==candidates
        else:
            assert case['phenotype']=='socket_contacts_exposed'
            assert row['decision']=='reference_socket_exposure_observed' and row['candidate_records']==[]
        assert row['same_physical_wire_confirmed'] is False and row['electrical_continuity']=='not_assessed'
        assert row['physical_new_connections']==0 and row['mask_pixels_changed'] is False
        for anchor in scope['anchors']:
            pose=next(p for p in case['anchors'] if p['id']==anchor['id'])
            quad=np.array(row['endpoint_source_quads'][anchor['id']])
            homogeneous=np.column_stack((quad,np.ones(4)))
            back=homogeneous@np.array(pose['inspection_to_reference_local']).T
            back=back[:,:2]/back[:,2:]
            l,t,r,b=anchor['bbox_xyxy']
            assert np.allclose(back,np.array([[l,t],[r,t],[r,b],[l,b]]),atol=1e-7)
        checks.append(dict(id=case['id'],decision=row['decision'],native_candidates=len(candidates),
                           endpoint_coordinate_roundtrip='PASS'))
    assert source_pins()==protocol['mainline_pins']
    save(OUT/'audit_report.json',dict(status='PASS',checks=checks,report_sha256=digest(OUT/'report.json'),
        independent_native_flood_scalar_projection=True,reused_anchor_poses_and_socket_phenotypes=True,
        fresh_SAM=False,physical_identity_not_validated=True,not_field_accuracy=True))
    print('all5 native source replay, endpoint coordinates and claim boundaries PASS')


if __name__=='__main__':main()
