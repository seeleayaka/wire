"""One local job: new original SAM inference followed by unchanged bundle gate."""
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from run_prompt_contrast import save,verify
from run_paired_evidence import load_run
from run_review import verified_run
from analyze_mendeley_scope import render_all
from visible_bundle_relation import observe_bundle,compare_bundle
ROOT=Path(__file__).resolve().parents[2]
def review(evidence):
    protocol=json.loads((evidence/'protocol.json').read_text(encoding='utf-8'))
    inference=json.loads((evidence/'inference_report.json').read_text(encoding='utf-8'))
    preparation=json.loads((evidence/'preparation_report.json').read_text(encoding='utf-8'))
    assert inference['status']=='complete' and inference['protocol_sha256']==sha256(evidence/'protocol.json')
    verify(protocol['pins']);assert source_pins()==protocol['mainline_pins']
    scope=json.loads(Path(protocol['confirmed_scope_path']).read_text(encoding='utf-8'))
    out=ROOT/'artifacts/mendeley_typed_affine_bundle_review_20261006';out.mkdir(exist_ok=False);observations=[]
    for row in preparation['cases']:
        origin=row['original_source'];assert sha256(origin['path'])==origin['image_sha256']
        binding={k:origin[k] for k in ['image_sha256','image_size','coordinate_frame']};masks=[];context=row['crop_context']
        if row['sam_inference_requested']:
            actual=np.asarray(Image.open(origin['path']).convert('RGB').crop(context['crop_box_xyxy']))
            assert np.array_equal(actual,np.asarray(Image.open(context['source']['path']).convert('RGB')))
            for recipe in protocol['recipes']:
                run=evidence/row['id']/recipe;source,records=load_run(run);verified_run(run,source['image_path'])
                render_all(source,records,out/(row['id']+'_'+recipe))
                if recipe.endswith('_box'):
                    for r in records:masks.append(dict(raw=np.asarray(Image.open(r['source_mask_path']).convert('L')),record_id=r['record_id'],score=r['score'],recipe=recipe))
        observation=observe_bundle(scope,binding,row['anchors'],masks,row['phenotype'],
            translation=tuple(context['crop_box_xyxy'][:2]) if context else (0,0),sam_inventory_verified=row['sam_inference_requested'])
        observations.append(dict(id=row['id'],observation=observation))
    reference=observations[0]['observation'];cases=[dict(**c,comparison=compare_bundle(scope,reference,c['observation'])) for c in observations[1:]]
    verify(protocol['pins']);assert source_pins()==protocol['mainline_pins']
    save(out/'report.json',dict(status='complete',reference_observation=reference,cases=cases,
        fresh_SAM_encoders=inference['fresh_encoders'],fresh_SAM_decoders=inference['fresh_decoders'],
        native_mask_pixels_unchanged=True,bundle_policy_unchanged=True,visual_review_and_independent_audit='pending',
        electrical_connections_confirmed=0,posthoc_repeated_development_not_field_accuracy=True,production_unchanged=True,deployed=False))
def main():
    evidence=ROOT/'artifacts/mendeley_typed_affine_bundle_20261006'
    if (evidence/'pipeline_progress.json').exists():raise FileExistsError('already started, no duplicate')
    digest=sha256(__file__);save(evidence/'pipeline_contract.json',dict(pipeline_sha256=digest,no_retry=True,no_deploy=True))
    save(evidence/'pipeline_progress.json',dict(status='running',stage='fresh_SAM',pid=os.getpid()))
    with (evidence/'fresh_SAM.log').open('x',encoding='utf-8') as stream:
        result=subprocess.run(['E:/PythonProject10/runtime/sam3/.venv/Scripts/python.exe','-B',str(Path(__file__).with_name('run_mendeley_geometry.py')),
            '--protocol',str(evidence/'protocol.json')],cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1'),stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        save(evidence/'pipeline_progress.json',dict(status='failed',stage='fresh_SAM',exit_code=result.returncode));raise RuntimeError('SAM failed; no retry')
    save(evidence/'pipeline_progress.json',dict(status='running',stage='native_bundle_analysis',pid=os.getpid()))
    try:review(evidence)
    except BaseException as error:
        save(evidence/'pipeline_progress.json',dict(status='failed',stage='native_bundle_analysis',error=str(error)));raise
    assert sha256(__file__)==digest
    save(evidence/'pipeline_progress.json',dict(status='complete',actual_visual_review='pending',independent_audit='pending',deployed=False))
if __name__=='__main__':main()
