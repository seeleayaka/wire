"""Independent all30 native flood fill/scalar anchor-hit audit, no inference."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from run_prompt_contrast import save,verify,digest
from run_review import verified_run
from bundle_runtime_pins import source_pins
from audit_semantic_visible_bundle_native import flood_components
from audit_cable_socket_controls import replay_hits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/endpoint_pair_expanded30_v3_20261008'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p=read(OUT/'protocol.json');verify(p['pins'])
    scope=read(ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json')
    base=ROOT/'artifacts/mendeley_confirmed_bundle_batch_v2_20261006'
    recent=ROOT/'artifacts/mendeley_semantic_visible_bundle_20261006'
    prepared={r['id']:r for r in read(base/'preparation_report.json')['cases']}
    prepared.update({r['id']:r for r in read(recent/'preparation_report.json')['cases']})
    report=read(OUT/'report.json')
    assert report['status']=='complete' and len(report['cases'])==30
    checks=[];total_masks=0;total_components=0
    for row in report['cases']:
        case=prepared[row['id']];context=case.get('crop_context')
        assert digest(row['source_path'])==row['source_binding']['image_sha256']
        candidates=[]
        if row['native_run_directory']:
            crop=context['crop_box_xyxy']
            pixels=np.asarray(Image.open(row['source_path']).convert('RGB').crop(crop))
            assert np.array_equal(pixels,np.asarray(Image.open(context['source']['path']).convert('RGB')))
            inventory=verified_run(row['native_run_directory'],context['source']['path'])
            assert len(inventory['paths'])==len(row['native_mask_audit'])
            for path,score,old in zip(inventory['paths'],inventory['scores'],row['native_mask_audit']):
                mask=np.asarray(Image.open(path).convert('L'))>0
                ys,xs=np.where(mask);h,w=mask.shape
                boundary=bool(len(xs) and (xs.min()<=1 or ys.min()<=1 or xs.max()>=w-2 or ys.max()>=h-2))
                parts=flood_components(mask)
                hits=replay_hits(parts,w,crop[:2],scope['anchors'],case['anchors'])
                assert boundary==old['boundary_truncated'] and float(score)==old['score']
                assert sorted((n,tuple(sorted(hit.items()))) for n,hit in hits)==sorted(
                    (c['pixel_count'],tuple(sorted(c['anchor_pixel_support'].items()))) for c in old['components'])
                total_masks+=1;total_components+=len(parts)
                coverage={a['id']:[i for i,(_,hit) in enumerate(hits,1) if hit[a['id']]>0] for a in scope['anchors']}
                if score>=.75 and not boundary and all(coverage.values()):
                    candidates.append(dict(record_id=path.stem,anchor_components=coverage))
        old_decision=row['automatic_topology_comparison']['decision']
        if old_decision=='visible_socket_attachment_change_supported':
            assert row['decision']=='reference_socket_exposure_observed'
            assert row['single_endpoint_observation_only'] is True
        elif row['socket_evidence_conflict']:
            assert row['decision']=='insufficient_endpoint_evidence' and row['candidate_records']==[]
        elif (row['native_inventory_verified'] and all(row['local_anchor_support'].values())
              and row['socket_phenotype_observed']=='mating_body_visible' and len(candidates)==1):
            assert row['decision']=='reference_endpoint_pair_candidate'
            assert [{k:c[k] for k in ['record_id','anchor_components']} for c in row['candidate_records']]==candidates
        else:
            assert row['decision']=='insufficient_endpoint_evidence'
        assert row['same_physical_wire_confirmed'] is False and row['physical_new_connections']==0
        assert row['electrical_continuity']=='not_assessed' and row['mask_pixels_changed'] is False
        checks.append(dict(id=row['id'],decision=row['decision'],native_same_instance_pair_candidates=len(candidates)))
    assert source_pins()==p['mainline_pins'];verify(p['pins'])
    save(OUT/'audit_report.json',dict(status='PASS',cases=checks,checked_cases=30,native_masks=total_masks,
        native_components=total_components,independent_flood_fill_scalar_projection=True,
        reused_pose_and_socket_states=True,report_sha256=digest(OUT/'report.json'),
        new_pair_candidates_are_not_physical_identity_GT=True,fresh_SAM=False,deployed=False))
    print('PASS all30:',total_masks,'native masks;',total_components,'original components')


if __name__=='__main__':main()
