"""Audit all fresh SAM pixels and classify a visible relation, not continuity."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from run_prompt_contrast import verify
from run_paired_evidence import load_run
from run_review import verified_run
from analyze_mendeley_scope import render_all
from visible_bundle_relation import observe_bundle, compare_bundle

ROOT=Path(__file__).resolve().parents[2]
def main():
    evidence=ROOT/'artifacts/mendeley_compound_bundle_pair_20261006'
    protocol=json.loads((evidence/'protocol.json').read_text(encoding='utf-8'))
    inference=json.loads((evidence/'inference_report.json').read_text(encoding='utf-8'))
    preparation=json.loads((evidence/'preparation_report.json').read_text(encoding='utf-8'))
    if inference['status']!='complete' or inference['protocol_sha256']!=sha256(evidence/'protocol.json'):raise ValueError('fresh SAM incomplete/drift')
    verify(protocol['pins'])
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E drift')
    scope=json.loads(Path(protocol['confirmed_scope_path']).read_text(encoding='utf-8'))
    out=ROOT/'artifacts/mendeley_compound_bundle_pair_review_20261006';out.mkdir(exist_ok=False)
    observations=[];counts={}
    for row in preparation['cases']:
        context=row['crop_context'];binding={k:row['original_source'][k] for k in ['image_sha256','image_size','coordinate_frame']}
        if sha256(row['original_source']['path'])!=binding['image_sha256']:raise ValueError('original drift')
        original=np.asarray(Image.open(row['original_source']['path']).convert('RGB').crop(context['crop_box_xyxy']))
        crop=np.asarray(Image.open(context['source']['path']).convert('RGB'))
        if not np.array_equal(original,crop):raise ValueError('SAM input is not exact original pixels')
        if not row['sam_inference_requested']:raise ValueError('both pair members require fresh SAM')
        masks=[]
        for recipe in protocol['recipes']:
            run=evidence/row['id']/recipe;origin,records=load_run(run)
            verified_run(run,origin['image_path'])
            render_all(origin,records,out/(row['id']+'_'+recipe))
            counts[row['id']+'_'+recipe]=len(records)
            if recipe=='cable_plus_reference_anatomy_box':
                for r in records:
                    raw=np.asarray(Image.open(r['source_mask_path']).convert('L'))
                    masks.append(dict(raw=raw,record_id=r['record_id'],score=r['score'],recipe=recipe))
        observation=observe_bundle(scope,binding,row['anchors'],masks,row['phenotype'],
            translation=tuple(context['crop_box_xyxy'][:2]),sam_inventory_verified=True)
        observations.append(dict(id=row['id'],observation=observation))
    comparison=compare_bundle(scope,observations[0]['observation'],observations[1]['observation'])
    verify(protocol['pins'])
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E drift')
    result=dict(status='complete',comparison=comparison,cases=observations,SAM_instances=counts,
        fresh_encoders=inference['fresh_encoders'],fresh_decoders=inference['fresh_decoders'],
        inference_seconds=inference['seconds'],mask_pixels_changed=False,
        electrical_connections_confirmed=0,deployed=False,mainline_unchanged=True,
        selected_posthoc_development_not_independent_validation=True,visual_review_pending=True,
        pins={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('visible_bundle_relation.py'),evidence/'protocol.json',evidence/'inference_report.json']})
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(comparison=comparison,SAM_instances=counts)))
if __name__=='__main__':main()
