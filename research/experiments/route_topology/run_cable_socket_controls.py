"""Fresh fixed controls then unchanged native observation; no automatic repair."""
import json
import os
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from run_prompt_contrast import save,verify
from run_paired_evidence import load_run
from run_review import verified_run
from visible_bundle_relation import observe_bundle

ROOT=Path(__file__).resolve().parents[2]

def main():
    out=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
    progress=out/'pipeline_progress.json'
    if progress.exists():raise FileExistsError('no duplicate or automatic retry')
    digest=sha256(__file__)
    save(out/'pipeline_contract.json',dict(worker_sha256=digest,no_retry=True,no_deploy=True))
    save(progress,dict(status='running',stage='fresh_SAM',pid=os.getpid()))
    try:
        with (out/'fresh_SAM.log').open('x',encoding='utf-8') as stream:
            result=subprocess.run(['E:/PythonProject10/runtime/sam3/.venv/Scripts/python.exe','-B',
                str(Path(__file__).with_name('run_mendeley_geometry.py')),'--protocol',str(out/'protocol.json')],
                cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1'),stdout=stream,stderr=subprocess.STDOUT)
        assert result.returncode==0,'fresh SAM failed; no retry'
        protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
        inference=json.loads((out/'inference_report.json').read_text(encoding='utf-8'))
        assert inference['status']=='complete' and inference['protocol_sha256']==sha256(out/'protocol.json')
        verify(protocol['pins']);assert source_pins()==protocol['mainline_pins']
        scope=json.loads(Path(protocol['confirmed_scope_path']).read_text(encoding='utf-8'))
        preparation=json.loads((out/'preparation_report.json').read_text(encoding='utf-8'))
        contract=json.loads((out/'experiment_contract.json').read_text(encoding='utf-8'))
        rows=[]
        for row in preparation['cases']:
            origin=row['original_source'];context=row['crop_context'];assert sha256(origin['path'])==origin['image_sha256']
            actual=np.asarray(Image.open(origin['path']).convert('RGB').crop(context['crop_box_xyxy']))
            assert np.array_equal(actual,np.asarray(Image.open(context['source']['path']).convert('RGB')))
            masks=[]
            for recipe in protocol['recipes']:
                run=out/row['id']/recipe;source,records=load_run(run);verified_run(run,source['image_path'])
                if recipe.endswith('_box'):
                    masks.extend(dict(raw=np.asarray(Image.open(r['source_mask_path']).convert('L')),record_id=r['record_id'],score=r['score'],recipe=recipe) for r in records)
            binding={k:origin[k] for k in ['image_sha256','image_size','coordinate_frame']}
            observation=observe_bundle(scope,binding,row['anchors'],masks,row['phenotype'],translation=tuple(context['crop_box_xyxy'][:2]),sam_inventory_verified=True)
            rows.append(dict(id=row['id'],observation=observation))
        counterexamples=[r['id'] for r in rows if r['id'] in contract['expected_exposed_ids'] and r['observation']['high_score_mask_touches_socket']]
        verify(protocol['pins']);assert source_pins()==protocol['mainline_pins'] and sha256(__file__)==digest
        save(out/'report.json',dict(status='complete',cases=rows,fresh_encoders=inference['fresh_encoders'],fresh_decoders=inference['fresh_decoders'],
            exposed_controls_with_high_socket_touch=counterexamples,
            broad_mask_touch_specificity_hypothesis='refuted_on_qualitative_controls' if counterexamples else 'not_refuted_in_these_controls',
            independent_native_audit_and_actual_visual_review='pending',decision_override=False,
            no_demo_extension=True,electrical_connections_confirmed=0,production_unchanged=True,deployed=False))
        save(progress,dict(status='complete',independent_audit='pending',visual_review='pending',deployed=False))
    except BaseException as error:
        save(progress,dict(status='failed',error=str(error),no_retry=True));raise

if __name__=='__main__':main()
