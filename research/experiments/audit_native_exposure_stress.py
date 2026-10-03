"""ALL78 existing validation photos, fixed exposure, fresh actual core workflow."""
import copy
import os
import shutil
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import metric
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,run_native_pose_review
from inspection_agent.paired_median_geometry import run_paired_median_review
from inspection_agent.optional_port_crop_review import SCENE
from paired_graph_hint_link import accepted_native_rows
from actual_port_native_rows import actual_native_rows
OUT=ROOT/'artifacts/native_exposure_stress_20261004'
PLAN=ROOT/'artifacts/native_exposure_stress_preregistration_20261004/PLAN.md'
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',
    YOLO_CONFIG_DIR=str(OUT/'config'),PYTHONPROJECT10_DINO_CACHE=str(OUT/'dino_cache'))

def main():
    if OUT.exists():raise FileExistsError('Preserve exposure robustness audit')
    import torch
    import cv2
    import numpy as np
    import psutil
    assert psutil.virtual_memory().available>6*2**30,'No parallel heavy inference'
    torch.set_num_threads(2);cv2.setNumThreads(2)
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation;app=gui.QApplication.instance() or gui.QApplication([])
    (OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    frozen=native_pose_runtime_fingerprint(REPO);started=time.monotonic();cases=[]
    reference=DATA/'images/train01/normal_073.JPG';assert sha(reference)==REFERENCE_SHA
    pins={str(p):sha(p) for p in (Path(__file__),PLAN,reference,
        Path(__file__).with_name('actual_port_native_rows.py'),Path(__file__).with_name('paired_graph_hint_link.py'))}
    selections=[(stage,row) for stage in ('inner','outer') for row in load(BASE/stage/'report.json')['cases']];assert len(selections)==78
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,exposure_multiplier=.85,all_inner48_outer30=True,
        no_training=True,no_GT_in_proposal_selection=True,no_geometric_transform=True,losslessPNG=True,
        synthetic_same_scene_stress_not_new_data=True,no_SAM_recomputation=True,field_accuracy=False))
    previous=gui.adaptive.robust.auto.base.OUT
    try:
        for i,(stage,item) in enumerate(selections):
            name=item['image'];folder=OUT/stage/Path(name).stem;folder.mkdir(parents=True)
            source=DATA/'images'/('val01' if stage=='outer' else 'train01')/name;original_sha=sha(source);pins[str(source)]=original_sha
            image=read_image(source);assert list(image.shape[:2])==[2736,3648]
            transformed=np.rint(image.astype(np.float32)*.85).clip(0,255).astype(np.uint8)
            augmented=folder/'exposure085.png';ok,encoded=cv2.imencode('.png',transformed);assert ok;encoded.tofile(augmented)
            assert np.array_equal(read_image(augmented),transformed);pins[str(augmented)]=sha(augmented)
            gui.adaptive.robust.auto.base.OUT=folder
            def progress(phase):save(OUT/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),completed=i,total=78,stage=stage,image=name,phase=phase))
            progress('fresh_exposure_SIFT_DINO');cv2.setRNGSeed(0)
            worker=gui.InitialReviewWorker(reference,augmented,[[.03,.04,.97,.96]]);payloads=[];worker.completed.connect(payloads.append);worker.run()
            assert len(payloads)==1 and payloads[0]['status']=='ready_for_sam3',payloads[0].get('error') if payloads else 'no payload'
            report=payloads[0]['report'];directory=Path(payloads[0]['output']);save(directory/'initial_report.json',report);protected=copy.deepcopy(report)
            assert report['image_fingerprints']['source_sha256']==pins[str(augmented)] and report['image_fingerprints']['reference_sha256']==REFERENCE_SHA
            progress('fresh_accepted_median_then_native')
            old=run_paired_median_review(report,project=REPO,median_enabled=True,paired_enabled=True,enabled=True,scene=SCENE,
                supplementary_enabled=True,student_enabled=True,feature_enabled=True,resolution_enabled=True)
            # Run only the new append branch against the actual fresh median;
            # otherwise the wrapper would duplicate its full inference.
            from inspection_agent.paired_native_pose import append_native_pose_review
            native=append_native_pose_review(report,old,project=REPO);assert report==protected
            save(directory/'accepted_median_ports.json',old);save(directory/'native_pose_ports.json',native)
            for key,value in old.items():
                if key=='supplementary_hints':assert native[key][:len(value)]==value
                else:assert native[key]==value
            existing=actual_native_rows(report,old)
            final_rows=actual_native_rows(report,native)
            assert final_rows[:len(existing)]==existing
            accepted=final_rows[len(existing):]
            targets=read_targets(stage,name,[2736,3648],item['label_sha256'],pins)
            row=dict(stage=stage,image=name,original_sha256=original_sha,augmented_sha256=pins[str(augmented)],
                median=metric(existing,targets),native=metric(existing+accepted,targets),added=len(accepted),
                upstream_status=old['status'],upstream_reason=old.get('fallback_reason'),
                native_reason=native.get('native_pose_policy',{}).get('fallback_reason'),
                safety_abstention=old['status']!='applied',report=str(directory/'initial_report.json'),result=str(directory/'native_pose_ports.json'))
            cases.append(row);save(OUT/'partial.json',dict(cases=cases));print(dict(completed=i+1,total=78,**row),flush=True)
            assert native_pose_runtime_fingerprint(REPO)==frozen and sha(source)==original_sha
        summary={stage:{version:{k:sum(row[version][k] for row in cases if row['stage']==stage) for k in ('tp','unmatched','fn','predictions','targets')}
            for version in ('median','native')} for stage in ('inner','outer')}
        assert all(sha(Path(p))==value for p,value in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen
        result=dict(status='complete',summary=summary,cases=cases,pins=pins,runtime=frozen,exposure_multiplier=.85,
            safety_abstentions=sum(row['safety_abstention'] for row in cases),no_training=True,no_deployment=True,
            no_SAM_recomputation=True,field_accuracy=False,seconds=round(time.monotonic()-started,2))
        save(OUT/'report.json',result);save(OUT/'progress.json',dict(status='complete',seconds=result['seconds']));print(dict(status='complete',summary=summary),flush=True)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error),completed=len(cases),pid=os.getpid()));raise
    finally:gui.adaptive.robust.auto.base.OUT=previous;app.processEvents()

if __name__=='__main__':main()
