"""Fresh isolated retry of the failed final upstream control; preserve the 19 completed cases."""
import os
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha
OUT=ROOT/'artifacts'/os.environ.get('WIRE_LAST_CONTROL_OUT','paired_pose_native_last_control_20261004')
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',
                  YOLO_CONFIG_DIR=str(OUT/'config'),PYTHONPROJECT10_DINO_CACHE=str(OUT/'dino_cache'))

def main():
    if OUT.exists():raise FileExistsError('Preserve isolated retry')
    import shutil
    import torch
    import cv2
    (OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    torch.set_num_threads(2);cv2.setNumThreads(2)
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation;app=gui.QApplication.instance() or gui.QApplication([])
    reference=DATA/'images/train01/normal_073.JPG';source=DATA/'images/train01/disconnected_030.JPG'
    save(OUT/'progress.json',dict(status='running',pid=os.getpid(),phase='fresh_initial_worker'))
    old=gui.adaptive.robust.auto.base.OUT;gui.adaptive.robust.auto.base.OUT=OUT
    try:
        cv2.setRNGSeed(0);worker=gui.InitialReviewWorker(reference,source,[[.03,.04,.97,.96]])
        payloads=[];worker.completed.connect(payloads.append);worker.run();app.processEvents()
        save(OUT/'upstream_payloads.json',payloads)
        assert len(payloads)==1,payloads
        payload=payloads[0]
        if payload['status']!='ready_for_sam3':
            save(OUT/'progress.json',dict(status='failed',payload=payload));raise RuntimeError('Fresh upstream: '+str(payload))
        from inspection_agent.paired_median_geometry import run_paired_median_review
        from inspection_agent.optional_port_crop_review import SCENE
        from paired_pose_native_final_backend import append_native_pose_review
        report=payload['report'];save(OUT/'initial_report.json',report)
        original=run_paired_median_review(report,project=REPO,median_enabled=True,paired_enabled=True,enabled=True,
             scene=SCENE,supplementary_enabled=True,student_enabled=True,feature_enabled=True,resolution_enabled=True)
        save(OUT/'accepted_median_ports.json',original)
        result=append_native_pose_review(report,original,project=REPO);save(OUT/'native_pose_ports.json',result)
        assert result['supplementary_hints']==original['supplementary_hints']
        assert not result['pose_geometry_policy'].get('fallback_reason')
        save(OUT/'report.json',dict(status='complete',stage='train',image=source.name,
             original_initial_worker_exception=True,retry_only_no_algorithm_changes=True,new_cues=0,
             source_sha256=sha(source),reference_sha256=sha(reference),initial_report=str(OUT/'initial_report.json'),
             evidence=str(OUT/'native_pose_ports.json')))
        save(OUT/'progress.json',dict(status='complete'))
    finally:gui.adaptive.robust.auto.base.OUT=old;app.processEvents()

if __name__=='__main__':main()
