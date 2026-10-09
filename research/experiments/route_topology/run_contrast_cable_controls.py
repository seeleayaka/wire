"""One fixed photometric trial, then native-source checks and strict source gate."""
import json
import os
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from run_prompt_contrast import verify,save
from run_paired_evidence import load_run
from run_review import verified_run
from visible_bundle_relation import observe_bundle
from local_contrast_view import contrast_view
from audit_semantic_visible_bundle_native import flood_components
from audit_cable_socket_controls import replay_hits
from analyze_mendeley_scope import render_all

ROOT=Path(__file__).resolve().parents[2]

def main():
    out=ROOT/'artifacts/mendeley_contrast_cable_controls_20261007';progress=out/'pipeline_progress.json'
    if progress.exists():raise FileExistsError('no duplicate or automatic retry')
    digest=sha256(__file__);save(progress,dict(status='running',stage='fresh_SAM',pid=os.getpid()))
    try:
        with (out/'fresh_SAM.log').open('x',encoding='utf-8') as stream:
            result=subprocess.run(['E:/PythonProject10/runtime/sam3/.venv/Scripts/python.exe','-B',
                str(Path(__file__).with_name('run_mendeley_geometry.py')),'--protocol',str(out/'protocol.json')],
                cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1'),stdout=stream,stderr=subprocess.STDOUT)
        assert result.returncode==0,'fresh SAM failed; no retry'
        protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
        inference=json.loads((out/'inference_report.json').read_text(encoding='utf-8'))
        assert inference['status']=='complete' and inference['protocol_sha256']==sha256(out/'protocol.json')
        contract=json.loads((out/'experiment_contract.json').read_text(encoding='utf-8'))
        baseline=json.loads(Path(contract['baseline_report']).read_text(encoding='utf-8'))
        before={r['id']:r['observation'] for r in baseline['cases']}
        scope=json.loads(Path(protocol['confirmed_scope_path']).read_text(encoding='utf-8'))
        prepared=json.loads((out/'preparation_report.json').read_text(encoding='utf-8'))['cases']
        verify(protocol['pins']);assert source_pins()==protocol['mainline_pins']
        rows=[];checks=[];socket=next(a['id'] for a in scope['anchors'] if a['kind']=='wire_entry_socket')
        for row in prepared:
            origin=row['original_source'];context=row['crop_context'];crop=context['crop_box_xyxy']
            assert sha256(origin['path'])==origin['image_sha256']
            native=np.asarray(Image.open(origin['path']).convert('RGB').crop(crop))
            assert np.array_equal(native,np.asarray(Image.open(context['original_crop_path']).convert('RGB')))
            assert np.array_equal(contrast_view(native),np.asarray(Image.open(context['source']['path']).convert('RGB')))
            masks=[]
            for recipe in protocol['recipes']:
                run=out/row['id']/recipe;source,records=load_run(run);verified_run(run,source['image_path'])
                # Draw fresh predicted masks over original RGB, not enhanced RGB.
                render_all(dict(source,image_path=context['original_crop_path']),records,out/(row['id']+'_'+recipe+'_original_RGB'))
                if recipe.endswith('_box'):
                    masks.extend(dict(raw=np.asarray(Image.open(r['source_mask_path']).convert('L')),record_id=r['record_id'],score=r['score'],recipe=recipe) for r in records)
            binding={k:origin[k] for k in ['image_sha256','image_size','coordinate_frame']}
            view=observe_bundle(scope,binding,row['anchors'],masks,row['phenotype'],translation=tuple(crop[:2]),sam_inventory_verified=True)
            recorded={r['record_id']:r for r in view['mask_audit']};eligible=0;high=False;component_count=0
            for item in masks:
                raw=item['raw']>0;h,w=raw.shape;parts=flood_components(raw);component_count+=len(parts)
                ys,xs=np.where(raw)
                boundary=bool(len(xs) and (xs.min()<=1 or ys.min()<=1 or xs.max()>=w-2 or ys.max()>=h-2))
                old=recorded[item['record_id']];assert boundary==old['boundary_truncated']
                signatures=replay_hits(parts,w,crop[:2],scope['anchors'],row['anchors'])
                assert sorted((n,tuple(sorted(p.items()))) for n,p in signatures)==sorted((p['pixel_count'],tuple(sorted(p['anchor_pixel_support'].items()))) for p in old['components'])
                for n,p in signatures:
                    if item['score']>=.75 and not boundary and all(v>0 for v in p.values()):eligible+=1
                    if item['score']>=.75 and p[socket]>0:high=True
            assert eligible==len(view['eligible_native_components']) and high==view['high_score_mask_touches_socket']
            rows.append(dict(id=row['id'],observation=view))
            checks.append(dict(id=row['id'],components=component_count,two_anchor_components=eligible,high_socket_touch=high))
        visible=[r for r in rows if r['observation']['socket_state']=='mating_body_visible']
        gains=[r['id'] for r in visible if r['observation']['unique_native_bundle_observation_supported'] and not before[r['id']]['unique_native_bundle_observation_supported']]
        losses=[r['id'] for r in visible if not r['observation']['unique_native_bundle_observation_supported'] and before[r['id']]['unique_native_bundle_observation_supported']]
        conflicts=[r['id'] for r in rows if r['observation']['socket_state']=='socket_contacts_exposed' and r['observation']['high_score_mask_touches_socket']]
        passed=bool(gains and not losses and not conflicts)
        verify(protocol['pins']);assert source_pins()==protocol['mainline_pins'] and sha256(__file__)==digest
        save(out/'report.json',dict(status='complete',cases=rows,native_pixel_independent_replay=checks,
            source_gate_passed=passed,new_visible_source_gains=gains,old_source_losses=losses,
            exposed_high_mask_conflicts=conflicts,fresh_encoders=inference['fresh_encoders'],fresh_decoders=inference['fresh_decoders'],
            actual_original_RGB_visual_review='pending',independent_observer_count=1,
            original_pose_and_appearance_evidence_explicitly_shared=True,
            no_demo_extension=True,electrical_connections_confirmed=0,production_unchanged=True,deployed=False))
        save(progress,dict(status='complete',source_gate_passed=passed,actual_visual_review='pending',deployed=False))
    except BaseException as error:
        save(progress,dict(status='failed',error=str(error),no_retry=True));raise

if __name__=='__main__':main()
