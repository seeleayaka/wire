"""Frozen30 source-pixel/native-mask endpoint-pair replay, never fresh SAM."""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from core import sha256
from bundle_runtime_pins import source_pins
from run_review import verified_run,save
from run_paired_evidence import load_run
from run_prompt_contrast import verify
from visible_bundle_relation import observe_bundle,compare_bundle
from endpoint_pair_candidate_v2 import compare_endpoint_pair
from run_endpoint_pair_trial import render,endpoint_quads

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/endpoint_pair_expanded30_20261008'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    begun=time.monotonic()
    base=ROOT/'artifacts/mendeley_confirmed_bundle_batch_v2_20261006'
    recent=ROOT/'artifacts/mendeley_semantic_visible_bundle_20261006'
    original_path=ROOT/'artifacts/mendeley_semantic_bundle_consolidated_20261006/report.json'
    original=read(original_path)
    assert original['status']=='complete' and len(original['cases'])==30
    verify(original['inputs'])
    before=source_pins()
    protocol=read(base/'protocol.json')
    assert before==protocol['mainline_pins']
    scope=read(protocol['confirmed_scope_path'])
    rows={r['id']:(base,r) for r in read(base/'preparation_report.json')['cases']}
    rows.update({r['id']:(recent,r) for r in read(recent/'preparation_report.json')['cases']})
    pins={str(p):sha256(p) for p in [original_path,base/'protocol.json',base/'preparation_report.json',
        recent/'protocol.json',recent/'preparation_report.json',Path(protocol['confirmed_scope_path']),
        Path(__file__),Path(__file__).with_name('endpoint_pair_candidate.py'),
        Path(__file__).with_name('endpoint_pair_candidate_v2.py'),Path(__file__).with_name('visible_bundle_relation.py')]}
    OUT.mkdir(exist_ok=False)
    save(OUT/'protocol.json',dict(pins=pins,mainline_pins=before,planned_cases=30,
        fresh_SAM=False,reused_masks_local_poses_socket_states=True,no_threshold_change=True,
        no_deployment=True,no_physical_identity_claim=True))
    def observe(identity):
        folder,row=rows[identity];source=row['original_source']
        assert sha256(source['path'])==source['image_sha256']
        binding={k:source[k] for k in ['image_sha256','image_size','coordinate_frame']}
        context=row.get('crop_context');masks=[];verified=False;run=None
        if row['sam_inference_requested']:
            original_crop=np.asarray(Image.open(source['path']).convert('RGB').crop(context['crop_box_xyxy']))
            assert np.array_equal(original_crop,np.asarray(Image.open(context['source']['path']).convert('RGB')))
            run=folder/identity/'cable_plus_reference_anatomy_box'
            origin,records=load_run(run)
            verified_run(run,origin['image_path'])
            for record in records:
                raw=np.asarray(Image.open(record['source_mask_path']).convert('L'))
                masks.append(dict(raw=raw,record_id=record['record_id'],score=record['score'],recipe='cable_plus_reference_anatomy_box'))
            verified=True
        view=observe_bundle(scope,binding,row['anchors'],masks,row['phenotype'],
            translation=tuple(context['crop_box_xyxy'][:2]) if context else (0,0),sam_inventory_verified=verified)
        view['_raw_masks']=masks
        return view,row,run
    reference,_,_=observe('reference')
    results=[]
    for previous in original['cases']:
        identity=previous['id'];view,row,run=observe(identity)
        baseline=compare_bundle(scope,reference,view)
        assert baseline==previous['comparison'], 'original topology drift '+identity
        result=compare_endpoint_pair(scope,reference,view)
        assert result['automatic_topology_comparison']==baseline
        result.update(id=identity,source_binding=view['source_binding'],source_path=row['original_source']['path'],
            crop_box_xyxy=row['crop_context']['crop_box_xyxy'] if row.get('crop_context') else None,
            native_run_directory=str(run) if run else None,
            native_inventory_verified=view['sam_inventory_verified'],
            local_anchor_support=view['local_anchor_proposals_supported'],
            native_mask_audit=view['mask_audit'],endpoint_source_quads=endpoint_quads(scope,row),
            newly_proposed_vs_old_unknown=(baseline['decision']=='insufficient_evidence'
                                          and result['decision']=='reference_endpoint_pair_candidate'))
        if row.get('crop_context'):
            render(scope,row,view,result,OUT/(identity+'.png'))
        results.append(result)
        save(OUT/'progress.json',dict(status='replaying',completed=len(results),total=30))
    counts={name:sum(r['decision']==name for r in results) for name in sorted({r['decision'] for r in results})}
    losses=[r['id'] for r in results if r['automatic_topology_comparison']['decision']=='same_visible_bundle_attachment_supported'
            and r['decision']!='reference_endpoint_pair_candidate']
    unsafe=[r['id'] for r in results if r['decision']=='reference_endpoint_pair_candidate' and
            (r['socket_phenotype_observed']!='mating_body_visible' or not all(r['local_anchor_support'].values()))]
    verify(pins)
    assert source_pins()==before and losses==[] and unsafe==[]
    report=dict(status='complete',cases=results,decision_counts=counts,
        new_candidate_ids=[r['id'] for r in results if r['newly_proposed_vs_old_unknown']],
        old_visible_support_losses=losses,unsafe_endpoint_candidates=unsafe,
        original_topology_results_unchanged=True,original_native_masks_unchanged=True,
        fresh_SAM_calls=0,new_confirmed_wire_identities=0,new_confirmed_electrical_connections=0,
        independent_audit='pending',actual_visual_review='pending',seconds=time.monotonic()-begun,
        reused_development_material_not_field_accuracy=True,deployed=False)
    save(OUT/'report.json',report)
    save(OUT/'progress.json',dict(status='complete',completed=30,total=30))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False))


if __name__=='__main__':main()
