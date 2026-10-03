"""Fresh ALL new-cue sources, unmodified original reference/ROI/warp safety."""
import copy
import os
import shutil
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import metric,matches
from inspection_agent.paired_median_geometry import run_paired_median_review,median_runtime_fingerprint
from inspection_agent.optional_port_crop_review import SCENE
from inspection_agent.context_port_recheck import render_consensus_overlay
from paired_pose_final_gate_backend import append_pose_review
from paired_graph_hint_link import accepted_native_rows
from paired_geometry_live_contract import validate_upstream
from verify_paired_geometry_live import parity
SOURCE = ROOT / 'artifacts/paired_pose_search_20261003'
MEDIAN = ROOT / 'artifacts/paired_median_current_head_20261003'
OUT = ROOT / 'artifacts/paired_pose_final_gates_20261003'
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',
    YOLO_CONFIG_DIR=str(OUT / 'config'),PYTHONPROJECT10_DINO_CACHE=str(OUT / 'dino_cache'))


def main():
    if OUT.exists(): raise FileExistsError('Preserve actual final-gate experiment')
    import psutil
    assert psutil.virtual_memory().available > 4*2**30, 'Insufficient available memory; preserve other jobs'
    frozen = median_runtime_fingerprint(REPO); source_report = load(SOURCE / 'report.json'); assert source_report['status'] == 'rejected'
    source_stage = load(SOURCE / 'train/report.json'); prospective = [r for r in source_stage['cases'] if r['additions']]
    assert len(prospective) == 4
    pins = dict(source_report['pins']); assert {p:sha(Path(p)) for p in pins} == pins
    for path in (Path(__file__),Path(__file__).with_name('paired_pose_final_gate_backend.py'),
        ROOT / 'artifacts/paired_pose_final_gate_preregistration_20261003/PLAN.md',SOURCE / 'report.json',SOURCE / 'train/report.json'):
        pins[str(path)] = sha(path)
    categories = [('train',r['image'],'ALL_prospective_pose_additions') for r in prospective]
    entries = {stage:{r['image']:r for r in load(BASE / stage / 'report.json')['cases']} for stage in ('train','inner','outer')}
    for stage in entries:
        categories.append((stage,sorted(n for n in entries[stage] if n.startswith('normal_'))[0],'first_normal_safety_control'))
    reference = DATA / 'images/train01/normal_073.JPG'; pins[str(reference)] = sha(reference)
    (OUT / 'config/Ultralytics').mkdir(parents=True); shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT / 'config/Ultralytics/Arial.ttf')
    save(OUT / 'protocol.json',dict(pins=pins,median_runtime_fingerprint=frozen,categories=categories,
        no_changed_reference_ROI_or_warp_gate=True,actual_initial_and_accepted_median=True,
        no_SAM_concurrent=True,no_deployment=True,field_accuracy=False))
    import torch
    import cv2
    import numpy as np
    torch.set_num_threads(2); cv2.setNumThreads(2)
    import assembly_auto_review_dino_v2 as entry_module
    gui = entry_module.implementation; app = gui.QApplication.instance() or gui.QApplication([])
    before_output = gui.adaptive.robust.auto.base.OUT; started = time.monotonic(); cases = []; overrides = {}
    try:
        for index,(stage,name,reason) in enumerate(categories):
            folder = OUT / stage / Path(name).stem; folder.mkdir(parents=True); gui.adaptive.robust.auto.base.OUT = folder
            source = DATA / 'images' / ('val01' if stage=='outer' else 'train01') / name; pins[str(source)] = sha(source)
            cached_path = SOURCE / stage / (Path(name).stem+'_predictions.json') if stage=='train' else None
            cached = load(cached_path) if cached_path and cached_path.exists() else None
            expected_pose = cached['trial']['paired_semantic_additions'] if cached else []
            def progress(phase):
                save(OUT / 'progress.json',dict(status='running',pid=os.getpid(),stage=stage,image=name,completed=index,total=len(categories),phase=phase,seconds=round(time.monotonic()-started,2)))
            progress('fresh_initial_SIFT_DINO'); cv2.setRNGSeed(0)
            worker = gui.InitialReviewWorker(reference,source,[[.03,.04,.97,.96]]); payloads = []; worker.completed.connect(payloads.append); worker.run()
            assert len(payloads)==1 and payloads[0]['status']=='ready_for_sam3'
            payload = payloads[0]; report = payload['report']; directory = Path(payload['output']); save(directory / 'initial_report.json',report)
            protected = copy.deepcopy(report)
            progress('actual_accepted_median_baseline')
            original = run_paired_median_review(report,project=REPO,median_enabled=True,paired_enabled=True,enabled=True,
                scene=SCENE,supplementary_enabled=True,student_enabled=True,feature_enabled=True,resolution_enabled=True)
            upstream = validate_upstream(report,original,expected_pose,normal_control=reason=='first_normal_safety_control')
            save(directory / 'accepted_median_ports.json',original)
            fixed_path = MEDIAN / stage / (Path(name).stem+'_predictions.json'); pins[str(fixed_path)] = sha(fixed_path)
            fixed = load(fixed_path)['trial']
            current_evidence = original.get('median_geometry_evidence') or original.get('paired_geometry_evidence')
            if current_evidence: parity(fixed['all_predictions'],current_evidence['native']['all_predictions'])
            elif expected_pose: raise AssertionError('Prospective new cue cannot skip actual native baseline')
            progress('pose_source_and_unchanged_reference_ROI_gates')
            output = append_pose_review(report,original,project=REPO); assert report == protected
            policy = output['pose_geometry_policy']; assert not policy.get('fallback_reason'), policy
            evidence = output.get('pose_geometry_evidence'); added_hints = output['supplementary_hints'][len(original['supplementary_hints']):]
            accepted = []
            if evidence:
                parity(fixed['all_predictions'],evidence['native_current']['all_predictions'])
                parity(expected_pose,evidence['native']['paired_semantic_additions'])
                accepted = accepted_native_rows(evidence['native']['paired_semantic_additions'],added_hints,
                    np.asarray(report['alignment']['source_to_reference_homography']),[2736,3648],[2736,3648])
            elif expected_pose: raise AssertionError('Prospective new cue silently lacked final-gate evidence')
            save(directory / 'pose_ports.json',output)
            render_consensus_overlay(directory / 'aligned.jpg',output,directory / 'pose_overlay.jpg')
            for key,value in original.items():
                if key=='supplementary_hints': assert output[key][:len(value)]==value
                else: assert output[key]==value
            selected = fixed['all_predictions']+accepted
            targets = read_targets(stage,name,[2736,3648],entries[stage][name]['label_sha256'],pins)
            oh,nh = matches(fixed['all_predictions'],targets)[0],matches(selected,targets)[0]
            row = dict(stage=stage,image=name,reason=reason,upstream=upstream,raw_pose_cues=len(expected_pose),accepted=len(accepted),
                current=metric(fixed['all_predictions'],targets),trial=metric(selected,targets),gained=sorted(nh-oh),lost=sorted(oh-nh),
                report=str(directory / 'initial_report.json'),evidence=str(directory / 'pose_ports.json'),overlay=str(directory / 'pose_overlay.jpg'))
            cases.append(row); save(OUT / 'partial.json',dict(cases=cases)); print(str(row),flush=True)
            if stage=='train': overrides[name]=row
            if reason=='first_normal_safety_control': assert not selected
        rows = []
        for row in source_stage['cases']:
            updated = overrides.get(row['image'])
            rows.append(dict(image=row['image'],current=row['current'],trial=updated['trial'] if updated else row['current'],
                gained=updated['gained'] if updated else [],lost=updated['lost'] if updated else []))
        totals = {v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
        assert totals['current']==source_stage['summary']['current']
        normal = sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
        qualifies = totals['trial']['tp']>289 and totals['trial']['unmatched']<=4 and not any(r['lost'] for r in rows) and normal==0
        assert {p:sha(Path(p)) for p in pins}==pins and median_runtime_fingerprint(REPO)==frozen
        result = dict(status='complete',qualifies_final_train_gate=qualifies,summary=totals,cases=cases,all192_cases=rows,pins=pins,
            fresh_all_prospective_source_cues=True,safety_abstentions=sum(r['upstream']!='applied' for r in cases),
            requires_inner_outer_source_and_final_gates_if_pass=True,sam_pending=True,no_deployment=True,field_accuracy=False,seconds=round(time.monotonic()-started,2))
        save(OUT / 'report.json',result); save(OUT / 'progress.json',dict(status='complete',qualifies=qualifies,summary=totals))
    except BaseException as error:
        save(OUT / 'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error))); raise
    finally:
        gui.adaptive.robust.auto.base.OUT = before_output; app.processEvents()


if __name__ == '__main__': main()
