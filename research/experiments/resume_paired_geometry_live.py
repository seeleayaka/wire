"""Pin-verified completed15 plus fresh3 normal controls, not an18-fresh rerun."""
import copy
import os
import shutil
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from verify_paired_geometry_live import OUT as OLD,parity,TRAINED,ROOT,REPO,DATA,BASE,load,save,sha,read_targets,matches,metric,accepted_native_rows,SCENE
from paired_geometry_live_contract import validate_upstream
from paired_port_geometry_backend import append_geometry_review
from inspection_agent.resolution_loose_plug_support import run_resolution_plug_review,resolution_runtime_fingerprint
from inspection_agent.context_port_recheck import render_consensus_overlay
OUT=ROOT/'artifacts/paired_geometry_live_resume_20261003'
os.environ.update(YOLO_CONFIG_DIR=str(OUT/'yolo_config'),PYTHONPROJECT10_DINO_CACHE=str(OUT/'dino_cache'))


def main():
    if OUT.exists():raise FileExistsError('Preserve continuation evidence')
    protocol=load(OLD/'protocol.json');pins=dict(protocol['pins']);frozen=protocol['runtime_fingerprint']
    assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
    records=load(OLD/'partial.json')['cases'];assert len(records)==15
    categories=protocol['categories'];assert [(r['stage'],r['image']) for r in records]==[(r['stage'],r['image']) for r in categories[:15]]
    resumed_pins={str(p):sha(p) for p in (OLD/'protocol.json',OLD/'partial.json',OLD/'progress.json',Path(__file__),Path(__file__).with_name('paired_geometry_live_contract.py'))}
    entries={s:{r['image']:r for r in load(BASE/s/'report.json')['cases']} for s in ('train','inner','outer')}
    import numpy as np
    for row in records:
        saved=load((TRAINED/'full_train' if row['stage']=='train' else TRAINED/'holdouts'/row['stage'])/(Path(row['image']).stem+'_predictions.json'))
        directory=Path(row['initial_report']).parent
        for path in (Path(row['initial_report']),Path(row['evidence']),directory/'current_v3_ports.json',Path(row['overlay'])):resumed_pins[str(path)]=sha(path)
        report=load(Path(row['initial_report']));original=load(directory/'current_v3_ports.json');output=load(Path(row['evidence']))
        assert original['status']=='applied' and not output['paired_geometry_policy'].get('fallback_reason')
        for key in ('parents','existing_hints','rescue_hints'):assert output[key]==original[key]
        assert output['supplementary_hints'][:len(original['supplementary_hints'])]==original['supplementary_hints']
        evidence=output.get('paired_geometry_evidence');accepted=[]
        if evidence:
            parity(saved['current']['all_predictions'],evidence['native_current']['all_predictions'])
            parity(saved['trial']['paired_semantic_additions'],evidence['native']['paired_semantic_additions'])
            accepted=accepted_native_rows(evidence['native']['paired_semantic_additions'],output['supplementary_hints'][len(original['supplementary_hints']):],np.asarray(report['alignment']['source_to_reference_homography']),[2736,3648],[2736,3648])
        else:assert not saved['trial']['paired_semantic_additions']
        targets=read_targets(row['stage'],row['image'],[2736,3648],entries[row['stage']][row['image']]['label_sha256'],pins)
        assert row['current']==metric(saved['current']['all_predictions'],targets)
        assert row['trial']==metric(saved['current']['all_predictions']+accepted,targets)
        assert not row['lost'] and row['trial']['unmatched']<=row['current']['unmatched']
    OUT.mkdir();(OUT/'yolo_config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'yolo_config/Ultralytics/Arial.ttf')
    save(OUT/'protocol.json',dict(original_protocol=protocol,continuation_pins=resumed_pins,
        completed15_outputs_rechecked=True,fresh_normal_controls=categories[15:],original_failure_preserved=True,
        explicit_local_alignment_abstention_not_recognition=True,no_gate_or_model_change=True,sam_pending=True,field_accuracy=False))
    import torch
    torch.set_num_threads(4)
    import cv2
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation;app=gui.QApplication.instance() or gui.QApplication([])
    previous=gui.adaptive.robust.auto.base.OUT;started=time.monotonic()
    try:
        for index,category in enumerate(categories[15:],15):
            stage,name=category['stage'],category['image'];target=OUT/stage/Path(name).stem;target.mkdir(parents=True)
            gui.adaptive.robust.auto.base.OUT=target
            save(OUT/'progress.json',dict(status='running',pid=os.getpid(),completed=index,total=18,image=name,stage=stage,phase='fresh_initial_normal_control'))
            source=DATA/'images'/('val01' if stage=='outer' else 'train01')/name;reference=DATA/'images/train01/normal_073.JPG'
            cv2.setRNGSeed(0);worker=gui.InitialReviewWorker(reference,source,[[.03,.04,.97,.96]])
            payloads=[];worker.completed.connect(payloads.append);worker.run();assert len(payloads)==1
            payload=payloads[0];assert payload['status']=='ready_for_sam3',payload['status']
            report=payload['report'];directory=Path(payload['output']);save(directory/'initial_report.json',report);protected=copy.deepcopy(report)
            original=run_resolution_plug_review(report,project=REPO,enabled=True,scene=SCENE,supplementary_enabled=True,student_enabled=True,feature_enabled=True,resolution_enabled=True)
            saved=load((TRAINED/'full_train' if stage=='train' else TRAINED/'holdouts'/stage)/(Path(name).stem+'_predictions.json'))
            upstream=validate_upstream(report,original,saved['trial']['paired_semantic_additions'],normal_control=True)
            save(directory/'current_v3_ports.json',original);output=append_geometry_review(report,original,project=REPO)
            save(directory/'paired_geometry_ports.json',output)
            assert report==protected and not output['paired_geometry_policy'].get('fallback_reason')
            for key in original:assert output[key]==original[key]
            assert not original['rescue_hints'] and not original['supplementary_hints']
            assert not saved['current']['all_predictions'] and not saved['trial']['paired_semantic_additions']
            render_consensus_overlay(directory/'aligned.jpg',output,directory/'paired_geometry_overlay.jpg')
            targets=read_targets(stage,name,[2736,3648],entries[stage][name]['label_sha256'],pins);assert not targets
            row=dict(stage=stage,image=name,reasons=category['reasons'],upstream_status=original['status'],upstream_path=upstream,
                current=metric([],targets),trial=metric([],targets),gained=[],lost=[],expected_candidates=0,accepted=0,
                current_hint_count=0,new_hint_count=0,initial_report=str(directory/'initial_report.json'),
                evidence=str(directory/'paired_geometry_ports.json'),overlay=str(directory/'paired_geometry_overlay.jpg'),sam_pending=True)
            records.append(row);save(OUT/'partial.json',dict(cases=records));print(str(row),flush=True)
        assert {p:sha(Path(p)) for p in pins}==pins and {p:sha(Path(p)) for p in resumed_pins}==resumed_pins
        assert resolution_runtime_fingerprint(REPO)==frozen
        gains={stage:sum(r['trial']['tp']-r['current']['tp'] for r in records if r['stage']==stage) for stage in ('train','inner','outer')}
        assert gains['train']>0 and gains['inner']>0
        save(OUT/'report.json',dict(status='complete',qualifies_live_diagnostic=True,cases=records,gains=gains,
            reused15_completed_actual_workflows=True,fresh3_normal_controls=True,original_failure='test_requires_applied_even_for_explicit_safety_abstention',
            safety_abstentions=sum(r.get('upstream_path')=='safety_abstention_local_alignment' for r in records),
            seconds=round(time.monotonic()-started,2),sam_pending=True,field_accuracy=False))
        save(OUT/'progress.json',dict(status='complete',gains=gains));print(str(gains),flush=True)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error),seconds=round(time.monotonic()-started,2)));raise
    finally:gui.adaptive.robust.auto.base.OUT=previous;app.processEvents()

if __name__=='__main__':main()
