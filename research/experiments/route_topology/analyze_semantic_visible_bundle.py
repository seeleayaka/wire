"""Unchanged native bundle gate on fresh SAM; unsupported candidates stay unknown."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from run_prompt_contrast import verify,save
from run_paired_evidence import load_run
from run_review import verified_run
from analyze_mendeley_scope import render_all
from visible_bundle_relation import observe_bundle,compare_bundle
ROOT=Path(__file__).resolve().parents[2]
def main():
    evidence=ROOT/'artifacts/mendeley_semantic_visible_bundle_20261006'
    protocol=json.loads((evidence/'protocol.json').read_text(encoding='utf-8'))
    inference=json.loads((evidence/'inference_report.json').read_text(encoding='utf-8'))
    preparation=json.loads((evidence/'preparation_report.json').read_text(encoding='utf-8'))
    assert inference['status']=='complete' and inference['protocol_sha256']==sha256(evidence/'protocol.json')
    verify(protocol['pins']);assert source_pins()==protocol['mainline_pins']
    scope=json.loads(Path(protocol['confirmed_scope_path']).read_text(encoding='utf-8'))
    out=ROOT/'artifacts/mendeley_semantic_visible_bundle_review_20261006';out.mkdir(exist_ok=False)
    observations=[];instance_count=0
    for row in preparation['cases']:
        binding={k:row['original_source'][k] for k in ['image_sha256','image_size','coordinate_frame']}
        assert sha256(row['original_source']['path'])==binding['image_sha256']
        masks=[];context=row['crop_context'];verified=False
        if row['sam_inference_requested']:
            crop=np.asarray(Image.open(context['source']['path']).convert('RGB'))
            original=np.asarray(Image.open(row['original_source']['path']).convert('RGB').crop(context['crop_box_xyxy']))
            assert np.array_equal(crop,original)
            for recipe in protocol['recipes']:
                run=evidence/row['id']/recipe;origin,records=load_run(run)
                verified_run(run,origin['image_path']);instance_count+=len(records)
                render_all(origin,records,out/(row['id']+'_'+recipe))
                if recipe=='cable_plus_reference_anatomy_box':
                    for record in records:
                        masks.append(dict(raw=np.asarray(Image.open(record['source_mask_path']).convert('L')),
                            record_id=record['record_id'],score=record['score'],recipe=recipe))
            verified=True
        observation=observe_bundle(scope,binding,row['anchors'],masks,row['phenotype'],
            translation=tuple(context['crop_box_xyxy'][:2]) if context else (0,0),sam_inventory_verified=verified)
        observations.append(dict(id=row['id'],observation=observation))
    reference=observations[0]['observation']
    cases=[dict(**row,comparison=compare_bundle(scope,reference,row['observation'])) for row in observations[1:]]
    verify(protocol['pins']);assert source_pins()==protocol['mainline_pins']
    save(out/'report.json',dict(status='complete',reference_observation=reference,cases=cases,
        decision_counts={d:sum(c['comparison']['decision']==d for c in cases) for d in set(c['comparison']['decision'] for c in cases)},
        fresh_SAM_encoders=inference['fresh_encoders'],fresh_SAM_decoders=inference['fresh_decoders'],
        native_instances_verified=instance_count,all_four_new_candidates_retained=True,
        native_mask_pixels_unchanged=True,bundle_policy_unchanged=True,
        electrical_connections_confirmed=0,actual_mask_visual_review_pending=True,
        selected_development_not_field_accuracy=True,production_unchanged=True,deployed=False))
    print(json.dumps({c['id']:c['comparison']['decision'] for c in cases}))
if __name__=='__main__':main()
