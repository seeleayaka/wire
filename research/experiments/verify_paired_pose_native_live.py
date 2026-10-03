"""Every new source through actual unchanged prefix and original safety gates."""
import copy
import os
import shutil
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import metric,matches
from inspection_agent.paired_median_geometry import run_paired_median_review,median_runtime_fingerprint
from inspection_agent.optional_port_crop_review import SCENE
from inspection_agent.context_port_recheck import render_consensus_overlay
from paired_pose_native_final_backend import append_native_pose_review,HEAD,HEAD_SHA
from paired_graph_hint_link import accepted_native_rows
from paired_geometry_live_contract import validate_upstream
from verify_paired_geometry_live import parity
SOURCE=ROOT/'artifacts/paired_pose_native_three_20261004'
AUDIT=ROOT/'artifacts/paired_pose_native_three_audit_20261004/report.json'
MEDIAN=ROOT/'artifacts/paired_median_current_head_20261003'
OUT=ROOT/'artifacts/paired_pose_native_live_20261004'
PLAN=ROOT/'artifacts/paired_pose_native_live_preregistration_20261004/PLAN.md'
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',
    YOLO_CONFIG_DIR=str(OUT/'config'),PYTHONPROJECT10_DINO_CACHE=str(OUT/'dino_cache'))


def main():
    if OUT.exists():raise FileExistsError('Preserve actual learned-native acceptance')
    import psutil
    assert psutil.virtual_memory().available>4*2**30,'Insufficient available memory'
    final=load(SOURCE/'report.json');assert final['status']=='source_pass_requires_reference_ROI_SAM_Qt'
    assert load(AUDIT)['status']=='complete';frozen=median_runtime_fingerprint(REPO);assert frozen==final['median_runtime_fingerprint']
    pins=dict(final['pins']);pins.update(load(AUDIT)['pins'])
    for path in (Path(__file__),Path(__file__).with_name('paired_pose_native_final_backend.py'),
        Path(__file__).with_name('paired_pose_final_gate_backend.py'),PLAN,HEAD,AUDIT,SOURCE/'report.json'):pins[str(path)]=sha(path)
    assert sha(HEAD)==HEAD_SHA and {p:sha(Path(p)) for p in pins}==pins
    categories={};entries={};source_stages={}
    def add(stage,name,reason):categories.setdefault((stage,name),[]).append(reason)
    for stage in ('train','inner','outer'):
        entries[stage]={r['image']:r for r in load(BASE/stage/'report.json')['cases']}
        folder=SOURCE/('full_train' if stage=='train' else stage);source_stages[stage]=load(folder/'report.json')
        for row in source_stages[stage]['cases']:
            if row['trial']['predictions']>row['current']['predictions']:add(stage,row['image'],'ALL_new_native_pose_source')
        for row in load(MEDIAN/stage/'report.json')['cases']:
            if row.get('gained'):add(stage,row['image'],'ALL_previous_median_gain_control')
        add(stage,sorted(n for n in entries[stage] if n.startswith('normal_'))[0],'first_normal_safety_control')
    rejected=load(ROOT/'artifacts/paired_pose_final_gates_20261003/report.json')
    for row in rejected['cases']:
        if row['trial']['unmatched']>row['current']['unmatched']:add(row['stage'],row['image'],'ALL_previous_pose_false_cue_control')
    reference=DATA/'images/train01/normal_073.JPG';pins[str(reference)]=sha(reference)
    (OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    save(OUT/'protocol.json',dict(pins=pins,median_runtime_fingerprint=frozen,head_sha256=HEAD_SHA,
        categories=[dict(stage=s,image=n,reasons=r) for (s,n),r in categories.items()],
        all_actual_new_cue_sources=True,original_safety_gates_unchanged=True,no_SAM_concurrent=True,no_deployment=True,field_accuracy=False))
    import torch
    import cv2
    import numpy as np
    torch.set_num_threads(2);cv2.setNumThreads(2)
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation;app=gui.QApplication.instance() or gui.QApplication([])
    old_out=gui.adaptive.robust.auto.base.OUT;started=time.monotonic();cases=[];overrides={}
    try:
        for index,((stage,name),reasons) in enumerate(categories.items()):
            folder=OUT/stage/Path(name).stem;folder.mkdir(parents=True);gui.adaptive.robust.auto.base.OUT=folder
            source=DATA/'images'/('val01' if stage=='outer' else 'train01')/name;pins[str(source)]=sha(source)
            cached_path=SOURCE/('full_train' if stage=='train' else stage)/(Path(name).stem+'_predictions.json');pins[str(cached_path)]=sha(cached_path)
            cached=load(cached_path);expected=cached['trial']['paired_semantic_additions']
            def progress(phase):save(OUT/'progress.json',dict(status='running',pid=os.getpid(),stage=stage,image=name,completed=index,total=len(categories),phase=phase,seconds=round(time.monotonic()-started,2)))
            progress('fresh_initial_SIFT_DINO');cv2.setRNGSeed(0)
            worker=gui.InitialReviewWorker(reference,source,[[.03,.04,.97,.96]]);payloads=[];worker.completed.connect(payloads.append);worker.run()
            assert len(payloads)==1 and payloads[0]['status']=='ready_for_sam3'
            payload=payloads[0];report=payload['report'];directory=Path(payload['output']);save(directory/'initial_report.json',report);protected=copy.deepcopy(report)
            progress('actual_accepted_median_baseline')
            original=run_paired_median_review(report,project=REPO,median_enabled=True,paired_enabled=True,enabled=True,
                scene=SCENE,supplementary_enabled=True,student_enabled=True,feature_enabled=True,resolution_enabled=True)
            upstream=validate_upstream(report,original,expected,normal_control='first_normal_safety_control' in reasons)
            save(directory/'accepted_median_ports.json',original)
            fixed_path=MEDIAN/stage/(Path(name).stem+'_predictions.json');pins[str(fixed_path)]=sha(fixed_path);fixed=load(fixed_path)['trial']
            evidence=original.get('median_geometry_evidence') or original.get('paired_geometry_evidence')
            if evidence:parity(fixed['all_predictions'],evidence['native']['all_predictions'])
            elif expected:raise AssertionError('New cue missing actual prefix evidence')
            progress('native_learned_pose_and_original_reference_ROI_gates')
            output=append_native_pose_review(report,original,project=REPO);assert report==protected
            policy=output['pose_geometry_policy'];assert not policy.get('fallback_reason'),policy
            evidence=output.get('pose_geometry_evidence');hints=output['supplementary_hints'][len(original['supplementary_hints']):];accepted=[]
            if evidence:
                parity(fixed['all_predictions'],evidence['native_current']['all_predictions']);parity(expected,evidence['native']['paired_semantic_additions'])
                accepted=accepted_native_rows(evidence['native']['paired_semantic_additions'],hints,
                    np.asarray(report['alignment']['source_to_reference_homography']),[2736,3648],[2736,3648])
            elif expected:raise AssertionError('Prospective new cue missing actual final evidence')
            save(directory/'native_pose_ports.json',output);render_consensus_overlay(directory/'aligned.jpg',output,directory/'native_pose_overlay.jpg')
            for key,value in original.items():
                if key=='supplementary_hints':assert output[key][:len(value)]==value
                else:assert output[key]==value
            selected=fixed['all_predictions']+accepted;targets=read_targets(stage,name,[2736,3648],entries[stage][name]['label_sha256'],pins)
            old,new=matches(fixed['all_predictions'],targets)[0],matches(selected,targets)[0]
            row=dict(stage=stage,image=name,reasons=reasons,upstream=upstream,raw_new_cues=len(expected),accepted=len(accepted),
                current=metric(fixed['all_predictions'],targets),trial=metric(selected,targets),gained=sorted(new-old),lost=sorted(old-new),
                report=str(directory/'initial_report.json'),evidence=str(directory/'native_pose_ports.json'),overlay=str(directory/'native_pose_overlay.jpg'))
            cases.append(row);overrides[(stage,name)]=row;save(OUT/'partial.json',dict(cases=cases));print(str(row),flush=True)
            if 'first_normal_safety_control' in reasons:assert not selected
        stages={}
        for stage,source_stage in source_stages.items():
            rows=[]
            for row in source_stage['cases']:
                actual=overrides.get((stage,row['image']))
                rows.append(dict(image=row['image'],current=row['current'],trial=actual['trial'] if actual else row['current'],
                    gained=actual['gained'] if actual else [],lost=actual['lost'] if actual else []))
            summary={v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
            assert summary['current']==source_stage['summary']['current']
            gain=summary['trial']['tp']>summary['current']['tp'] if stage!='outer' else summary['trial']['tp']>=summary['current']['tp']
            normal=sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
            qualifies=gain and summary['trial']['unmatched']<=summary['current']['unmatched'] and not any(r['lost'] for r in rows) and normal==0
            stages[stage]=dict(qualifies=qualifies,summary=summary,cases=rows,normal_cues=normal)
        assert {p:sha(Path(p)) for p in pins}==pins and median_runtime_fingerprint(REPO)==frozen
        result=dict(status='complete',qualifies=all(r['qualifies'] for r in stages.values()),stages=stages,cases=cases,pins=pins,
            median_runtime_fingerprint=frozen,head_sha256=HEAD_SHA,safety_abstentions=sum(r['upstream']!='applied' for r in cases),
            all_prospective_new_sources_fresh=True,sam_pending=True,no_deployment=True,field_accuracy=False,seconds=round(time.monotonic()-started,2))
        save(OUT/'report.json',result);save(OUT/'progress.json',dict(status='complete',qualifies=result['qualifies']))
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise
    finally:gui.adaptive.robust.auto.base.OUT=old_out;app.processEvents()


if __name__=='__main__':main()
