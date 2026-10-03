"""Finite queued ALL78 original actual workflow for matched exposure diagnosis."""
import copy
import os
import shutil
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, DATA, load, save, sha, REFERENCE_SHA
from current_port_baseline_audit import BASE, read_targets
from audit_port_multiscale_acceptance import metric, matches
from actual_port_native_rows import actual_native_rows
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint, append_native_pose_review
from inspection_agent.paired_median_geometry import run_paired_median_review
from inspection_agent.optional_port_crop_review import SCENE
EXPOSURE = ROOT / 'artifacts/native_exposure_stress_20261004'
OUT = ROOT / 'artifacts/native_original_pairing_20261004'
PLAN = ROOT / 'artifacts/native_original_pairing_preregistration_20261004/PLAN.md'
os.environ.update(QT_QPA_PLATFORM='offscreen', HF_HUB_OFFLINE='1', YOLO_OFFLINE='True',
    YOLO_AUTOINSTALL='False', YOLO_CONFIG_DIR=str(OUT/'config'),
    PYTHONPROJECT10_DINO_CACHE=str(OUT/'dino_cache'))


def main():
    if OUT.exists(): raise FileExistsError('Preserve paired original robustness audit')
    import psutil
    OUT.mkdir()
    started = time.monotonic()
    frozen = native_pose_runtime_fingerprint(REPO)
    pins = {str(p):sha(p) for p in (Path(__file__), PLAN,
        Path(__file__).with_name('actual_port_native_rows.py'),
        Path(__file__).with_name('paired_graph_hint_link.py'))}
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,all78_originals=True,
        wait_for_existing_exposure_audit=True,maximum_wait_seconds=21600,
        no_training=True,no_SAM_recomputation=True,no_deployment=True,field_accuracy=False))
    try:
        while True:
            assert native_pose_runtime_fingerprint(REPO)==frozen
            status = load(EXPOSURE/'progress.json')
            if status.get('status')=='failed': raise RuntimeError('Upstream exposure audit failed; no duplicate retry')
            if status.get('status')=='complete' and psutil.virtual_memory().available>6*2**30: break
            if time.monotonic()-started>21600: raise TimeoutError('Bounded queued pairing wait expired')
            save(OUT/'progress.json',dict(status='queued',pid=os.getpid(),
                reason='await_existing_exposure_completion_and_memory',
                exposure_completed=status.get('completed'),
                wait_seconds=round(time.monotonic()-started,2),no_new_inference_started=True))
            time.sleep(30)
        import torch
        import cv2
        torch.set_num_threads(2)
        cv2.setNumThreads(2)
        (OUT/'config/Ultralytics').mkdir(parents=True)
        shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
        import assembly_auto_review_dino_v2 as entry
        gui=entry.implementation
        app=gui.QApplication.instance() or gui.QApplication([])
        exposure=load(EXPOSURE/'report.json')
        assert exposure['status']=='complete' and exposure['runtime']==frozen
        assert all(sha(Path(p))==value for p,value in exposure['pins'].items())
        pins[str(EXPOSURE/'report.json')]=sha(EXPOSURE/'report.json')
        reference=DATA/'images/train01/normal_073.JPG'
        assert sha(reference)==REFERENCE_SHA
        pins[str(reference)]=REFERENCE_SHA
        selections=[(stage,row) for stage in ('inner','outer') for row in load(BASE/stage/'report.json')['cases']]
        assert len(selections)==78 and [(s,r['image']) for s,r in selections]==[(r['stage'],r['image']) for r in exposure['cases']]
        previous=gui.adaptive.robust.auto.base.OUT
        cases=[]
        inference_started=time.monotonic()
        try:
            for i,((stage,item),dark) in enumerate(zip(selections,exposure['cases'])):
                name=item['image']
                source=DATA/'images'/('val01' if stage=='outer' else 'train01')/name
                source_sha=sha(source)
                assert source_sha==dark['original_sha256']
                pins[str(source)]=source_sha
                folder=OUT/stage/Path(name).stem
                folder.mkdir(parents=True)
                gui.adaptive.robust.auto.base.OUT=folder
                def progress(phase): save(OUT/'progress.json',dict(status='running',pid=os.getpid(),
                    completed=i,total=78,stage=stage,image=name,phase=phase,
                    seconds=round(time.monotonic()-inference_started,2)))
                progress('fresh_original_SIFT_DINO')
                cv2.setRNGSeed(0)
                worker=gui.InitialReviewWorker(reference,source,[[.03,.04,.97,.96]])
                payloads=[]
                worker.completed.connect(payloads.append)
                worker.run()
                assert len(payloads)==1 and payloads[0]['status']=='ready_for_sam3',payloads[0].get('error') if payloads else 'no payload'
                report=payloads[0]['report']
                directory=Path(payloads[0]['output'])
                save(directory/'initial_report.json',report)
                protected=copy.deepcopy(report)
                assert report['image_fingerprints']['source_sha256']==source_sha
                assert report['image_fingerprints']['reference_sha256']==REFERENCE_SHA
                progress('fresh_accepted_median_then_native')
                old=run_paired_median_review(report,project=REPO,median_enabled=True,paired_enabled=True,
                    enabled=True,scene=SCENE,supplementary_enabled=True,student_enabled=True,
                    feature_enabled=True,resolution_enabled=True)
                native=append_native_pose_review(report,old,project=REPO)
                assert report==protected
                for key,value in old.items():
                    if key=='supplementary_hints': assert native[key][:len(value)]==value
                    else: assert native[key]==value
                save(directory/'accepted_median_ports.json',old)
                save(directory/'native_pose_ports.json',native)
                original_rows=actual_native_rows(report,native)
                dark_report=load(Path(dark['report']))
                dark_result=load(Path(dark['result']))
                dark_rows=actual_native_rows(dark_report,dark_result)
                targets=read_targets(stage,name,[2736,3648],item['label_sha256'],pins)
                original_metric=metric(original_rows,targets)
                assert metric(dark_rows,targets)==dark['native']
                original_hits=matches(original_rows,targets)[0]
                dark_hits=matches(dark_rows,targets)[0]
                row=dict(stage=stage,image=name,source_sha256=source_sha,
                    original=original_metric,exposure=dark['native'],
                    lost_under_exposure=sorted(original_hits-dark_hits),
                    gained_under_exposure=sorted(dark_hits-original_hits),
                    original_abstention=old['status']!='applied',exposure_abstention=dark['safety_abstention'],
                    original_upstream_reason=old.get('fallback_reason'),
                    original_native_reason=native.get('native_pose_policy',{}).get('fallback_reason'),
                    report=str(directory/'initial_report.json'),result=str(directory/'native_pose_ports.json'))
                cases.append(row)
                save(OUT/'partial.json',dict(cases=cases))
                print(dict(completed=i+1,total=78,stage=stage,image=name,
                    original=original_metric,exposure=dark['native']),flush=True)
                assert native_pose_runtime_fingerprint(REPO)==frozen and sha(source)==source_sha
        finally:
            gui.adaptive.robust.auto.base.OUT=previous
        assert all(sha(Path(p))==value for p,value in pins.items())
        assert all(sha(Path(p))==value for p,value in exposure['pins'].items())
        assert native_pose_runtime_fingerprint(REPO)==frozen
        summary={stage:{version:{key:sum(row[version][key] for row in cases if row['stage']==stage)
            for key in ('tp','unmatched','fn','predictions','targets')}
            for version in ('original','exposure')} for stage in ('inner','outer')}
        result=dict(status='complete',summary=summary,cases=cases,pins=pins,runtime=frozen,
            lost_targets=sum(len(row['lost_under_exposure']) for row in cases),
            gained_targets=sum(len(row['gained_under_exposure']) for row in cases),
            synthetic_same_scene_not_independent_data=True,weak_GT_not_physical_truth=True,
            no_training=True,no_SAM_recomputation=True,no_deployment=True,field_accuracy=False,
            seconds=round(time.monotonic()-inference_started,2))
        save(OUT/'report.json',result)
        save(OUT/'progress.json',dict(status='complete',seconds=result['seconds'],summary=summary))
        print(dict(status='complete',summary=summary),flush=True)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)))
        raise


if __name__=='__main__':main()
