"""All gained and risk-control cohorts through real unchanged initial workflow."""
import copy
import os
import shutil
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_semantic_geometry import NEW as TRAINED
from prepare_paired_port_semantics import ROOT,REPO,DATA,PROPOSALS,load,save,sha,read_image
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import metric,matches
from paired_graph_hint_link import accepted_native_rows
from paired_port_geometry_backend import append_geometry_review,HEAD_SHA
from inspection_agent.optional_port_crop_review import SCENE
from inspection_agent.resolution_loose_plug_support import run_resolution_plug_review,resolution_runtime_fingerprint
from inspection_agent.context_port_recheck import render_consensus_overlay
OUT=ROOT/'artifacts/paired_geometry_live_20261003'
sys.path.insert(0,str(REPO/'prototype'))
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',
                  YOLO_CONFIG_DIR=str(OUT/'yolo_config'),PYTHONPROJECT10_DINO_CACHE=str(OUT/'dino_cache'))


def parity(old,new):
    import numpy as np
    if len(old)!=len(new):raise ValueError('fresh native count differs')
    for a,b in zip(old,new):
        if a['class_id']!=b['class_id'] or abs(a['confidence']-b['confidence'])>1e-6 or not np.allclose(a['box_xyxy'],b['box_xyxy'],rtol=0,atol=.001):
            raise ValueError('fresh native class/score/geometry differs')


def main():
    if OUT.exists():raise FileExistsError('Preserve actual workflow checks')
    categories={};snapshots={};entries={};pins={};frozen=resolution_runtime_fingerprint(REPO)
    for stage in ('train','inner','outer'):
        path=BASE/stage/'report.json';pins[str(path)]=sha(path);entries[stage]={row['image']:row for row in load(path)['cases']}
    def add(stage,name,reason):categories.setdefault((stage,name),[]).append(reason)
    for row in load(TRAINED/'full_train/report.json')['cases']:
        if row['additions']:add('train',row['image'],'ALL_full_head_training_candidate')
    for row in load(TRAINED/'holdouts/inner/report.json')['cases']:
        if row['additions']:add('inner',row['image'],'ALL_geometry_inner_candidate')
    for row in load(ROOT/'artifacts/port_extended_training_controls_20261003/report.json')['cases']:
        if row['metrics']['current']['predictions']>0:add('train',row['image'],'ALL_old_cue_control')
    for row in load(ROOT/'artifacts/paired_port_semantics_20261003/holdouts/inner/report.json')['cases']:
        if row['trial']['unmatched']>row['current']['unmatched']:add('inner',row['image'],'ALL_rejected_paired_inner_source')
    for row in load(ROOT/'artifacts/paired_semantic_committee_20261003/outer/report.json')['cases']:
        if row['trial']['unmatched']>row['current']['unmatched']:add('outer',row['image'],'ALL_rejected_committee_outer_source')
    for stage in ('train','inner','outer'):
        add(stage,sorted(name for name in entries[stage] if name.startswith('normal_'))[0],'first_normal_control')
    for stage,name in categories:
        path=(TRAINED/'full_train' if stage=='train' else TRAINED/'holdouts'/stage)/(Path(name).stem+'_predictions.json')
        pins[str(path)]=sha(path);snapshots[(stage,name)]=load(path)
        source=DATA/'images'/('val01' if stage=='outer' else 'train01')/name;pins[str(source)]=sha(source)
    reference=DATA/'images/train01/normal_073.JPG'
    for path in (Path(__file__),Path(__file__).with_name('paired_port_geometry_backend.py'),Path(__file__).with_name('paired_port_semantic_selection.py'),
                 Path(__file__).with_name('paired_port_semantics.py'),Path(__file__).with_name('port_semantic_verifier.py'),
                 TRAINED/'head_full/last_head.pt',reference,ROOT/'artifacts/paired_geometry_live_preregistration_20261003/PLAN.md'):
        pins[str(path)]=sha(path)
    assert sha(TRAINED/'head_full/last_head.pt')==HEAD_SHA
    OUT.mkdir();(OUT/'yolo_config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'yolo_config/Ultralytics/Arial.ttf')
    save(OUT/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,categories=[dict(stage=s,image=n,reasons=r) for (s,n),r in categories.items()],
        real_new_initial_registration_DINO_and_ports=True,all_candidate_and_risk_cohorts=True,source_GT_after_output_only=True,
        exact_native_hint_link=True,sam_pending=True,no_automatic_deployment=True,validation_reused=True,field_accuracy=False))
    import torch
    torch.set_num_threads(4)
    import cv2
    import numpy as np
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation;app=gui.QApplication.instance() or gui.QApplication([])
    previous=gui.adaptive.robust.auto.base.OUT;started=time.monotonic();records=[]
    def progress(**kw):save(OUT/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),**kw))
    try:
        for index,((stage,name),reasons) in enumerate(categories.items()):
            target=OUT/stage/Path(name).stem;target.mkdir(parents=True);gui.adaptive.robust.auto.base.OUT=target
            source=DATA/'images'/('val01' if stage=='outer' else 'train01')/name
            progress(stage=stage,image=name,completed=index,total=len(categories),phase='fresh_initial_registration_DINO')
            cv2.setRNGSeed(0);worker=gui.InitialReviewWorker(reference,source,[[.03,.04,.97,.96]])
            payloads=[];worker.completed.connect(payloads.append);worker.run();assert len(payloads)==1
            payload=payloads[0];assert payload['status']=='ready_for_sam3',payload['status']
            report=payload['report'];directory=Path(payload['output']);save(directory/'initial_report.json',report)
            protected=copy.deepcopy(report);progress(stage=stage,image=name,completed=index,total=len(categories),phase='real_current_V3')
            original=run_resolution_plug_review(report,project=REPO,enabled=True,scene=SCENE,supplementary_enabled=True,
                student_enabled=True,feature_enabled=True,resolution_enabled=True)
            save(directory/'current_v3_ports.json',original);assert original['status']=='applied'
            progress(stage=stage,image=name,completed=index,total=len(categories),phase='fresh_paired_geometry_and_reference')
            output=append_geometry_review(report,original,project=REPO);assert report==protected
            save(directory/'paired_geometry_ports.json',output)
            assert not output['paired_geometry_policy'].get('fallback_reason'),output['paired_geometry_policy']
            saved=snapshots[(stage,name)];current=saved['current'];expected=saved['trial']['paired_semantic_additions']
            evidence=output.get('paired_geometry_evidence');accepted=[]
            if evidence:
                parity(current['all_predictions'],evidence['native_current']['all_predictions'])
                parity(expected,evidence['native']['paired_semantic_additions'])
                new_hints=output['supplementary_hints'][len(original['supplementary_hints']):]
                accepted=accepted_native_rows(evidence['native']['paired_semantic_additions'],new_hints,
                    np.asarray(report['alignment']['source_to_reference_homography']),[2736,3648],[2736,3648])
            else:assert not expected,'Candidate source may not silently short circuit'
            render_consensus_overlay(directory/'aligned.jpg',output,directory/'paired_geometry_overlay.jpg')
            assert output['parents']==original['parents'] and output['existing_hints']==original['existing_hints'] and output['rescue_hints']==original['rescue_hints']
            assert output['supplementary_hints'][:len(original['supplementary_hints'])]==original['supplementary_hints']
            selected=current['all_predictions']+accepted
            targets=read_targets(stage,name,[2736,3648],entries[stage][name]['label_sha256'],pins)
            oh,nh=matches(current['all_predictions'],targets)[0],matches(selected,targets)[0]
            row=dict(stage=stage,image=name,reasons=reasons,current=metric(current['all_predictions'],targets),trial=metric(selected,targets),
                gained=sorted(nh-oh),lost=sorted(oh-nh),expected_candidates=len(expected),accepted=len(accepted),
                current_hint_count=len(original['rescue_hints'])+len(original['supplementary_hints']),
                new_hint_count=len(output['rescue_hints'])+len(output['supplementary_hints']),
                initial_report=str(directory/'initial_report.json'),evidence=str(directory/'paired_geometry_ports.json'),
                overlay=str(directory/'paired_geometry_overlay.jpg'),sam_pending=True)
            records.append(row);save(OUT/'partial.json',dict(cases=records));print(str(row),flush=True)
            assert not row['lost'] and row['trial']['unmatched']<=row['current']['unmatched']
            if name.startswith('normal_'):assert row['trial']['predictions']==0
            assert len(output['rescue_hints'])<=5 and len(output['supplementary_hints'])<=5
        gains={stage:sum(row['trial']['tp']-row['current']['tp'] for row in records if row['stage']==stage) for stage in ('train','inner','outer')}
        assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
        qualifies=gains['train']>0 and gains['inner']>0
        save(OUT/'report.json',dict(status='complete',qualifies_live_diagnostic=qualifies,cases=records,gains=gains,
            seconds=round(time.monotonic()-started,2),all_native_source_parity=True,sam_pending=True,no_automatic_deployment=True,field_accuracy=False))
        save(OUT/'progress.json',dict(status='complete',qualifies_live_diagnostic=qualifies,gains=gains));print(str(gains),flush=True)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error),seconds=round(time.monotonic()-started,2)));raise
    finally:gui.adaptive.robust.auto.base.OUT=previous;app.processEvents()


if __name__=='__main__':main()
