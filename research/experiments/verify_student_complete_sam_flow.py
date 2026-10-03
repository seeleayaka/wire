"""One training-selected gain: real SAM worker/final callback, then optional Qt port worker."""
import copy,json,os,shutil,sys,time
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');OUT=ROOT/'artifacts/student_complete_sam_flow_20261003'
SOURCE=ROOT/'artifacts/teacher_student_port_live_20261003/fresh_initial_train/20261003_010545'
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',YOLO_CONFIG_DIR=str(OUT/'config'))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    if OUT.exists():raise FileExistsError('Fresh output required')
    OUT.mkdir();(OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    import psutil,torch
    assert psutil.virtual_memory().available>6*2**30,'Insufficient free RAM for isolated SAM while training'
    torch.set_num_threads(2)
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation
    from inspection_agent.optional_port_crop_review import sha
    from audit_port_multiscale_acceptance import metric
    import cv2,numpy as np
    app=gui.QApplication.instance() or gui.QApplication([]);window=gui.DINOReview();report=load(SOURCE/'initial_report.json')
    protected={str(p):sha(p) for p in (SOURCE/'initial_report.json',SOURCE/'aligned.jpg',Path(report['inspection']),Path(report['reference']))}
    for name in ('aligned.jpg','valid_warp_mask.png','check_heatmap.jpg','anomaly_boxes.jpg','dino_anomaly_boxes.jpg'):shutil.copy2(SOURCE/name,OUT/name)
    settings=replace(gui.sam3_wire_fusion.load_settings(),cache_dir=OUT/'sam_cache',threads=2)
    save(OUT/'protocol.json',dict(training_selected_image=Path(report['inspection']).name,independent_accuracy=False,
        initial_alignment_dino_reused_sha_verified=True,fresh_isolated_sam_cache=True,new_sam_inference=True,
        sam_checkpoint_sha256=sha(settings.checkpoint),sam_settings=settings.report(),
        sam_thresholds_unchanged=True,threads_capped=2,protected=protected,automatic_fault_verdict=False))
    started=time.monotonic();save(OUT/'progress.json',dict(status='running',phase='fresh_sam',pid=os.getpid()))
    try:
        window.current_output=OUT;window.reference.setText(report['reference']);window.inspection.setText(report['inspection']);window._write_report(report)
        window._sam3_pending=dict(output=OUT,report=copy.deepcopy(report),dino_regions=copy.deepcopy(report['dino_review_regions']))
        worker=gui.Sam3FusionWorker(Path(report['reference']),OUT/'aligned.jpg',OUT/'valid_warp_mask.png',report['dino_review_regions'],OUT/'check_heatmap.jpg',OUT)
        payloads=[];worker.completed.connect(payloads.append)
        with patch.object(gui.sam3_wire_fusion,'load_settings',return_value=settings):worker.run()
        assert len(payloads)==1;save(OUT/'sam_result.json',payloads[0]);assert payloads[0]['status']=='ok',payloads[0].get('error')
        window._finish_sam3_fusion(payloads[0]);app.processEvents();final=load(OUT/'report.json')
        assert final['decision']!='sam3_fusion_running' and final['sam3_fusion']['status']=='ok'
        assert final['image_fingerprints']==report['image_fingerprints'] and final['analysis_check_rois']==report['analysis_check_rois']
        assert window.agent_task_path and window.agent_task_path.is_file()
        save(OUT/'before_optional_ports.json',final)
        save(OUT/'progress.json',dict(status='running',phase='real_port_button',pid=os.getpid()))
        window.port_scene_combo.setCurrentIndex(1);window.rescue_switch.setChecked(True);window.rescue_supplement_switch.setChecked(True);window.rescue_student_switch.setChecked(True);window._update_port_controls()
        assert window.rescue_button.isEnabled();window.rescue_button.click();assert window.rescue_worker is not None
        deadline=time.monotonic()+300
        while window.rescue_worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.02)
        assert window.rescue_worker is None,'owned port worker timeout';app.processEvents()
        saved=load(OUT/'report.json');ports=saved['independent_port_rescue'];assert ports['status']=='applied'
        assert ports['teacher_student_policy']['fallback_reason'] is None
        assert ports['teacher_student_policy']['added_hints']==1
        assert saved['review_regions']==final['review_regions'] and saved['decision']==final['decision']
        hints=ports['rescue_hints']+ports['supplementary_hints'];assert len(ports['rescue_hints'])<=5 and len(ports['supplementary_hints'])<=5
        assert not ports['automatic_fault_verdict'];assert {p:sha(Path(p)) for p in protected}==protected
        targets=[];matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64)
        label=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/labels/train01'/(Path(report['inspection']).stem+'.txt')
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,w,h=map(float,line.split())
            if cls not in (3,4):continue
            l,t,r,b=(cx-w/2)*3648,(cy-h/2)*2736,(cx+w/2)*3648,(cy+h/2)*2736
            points=cv2.perspectiveTransform(np.float32([[[l,t],[r,t],[r,b],[l,b]]]),matrix).reshape(-1,2)
            targets.append(dict(class_id=int(cls)-3,box=[points[:,0].min(),points[:,1].min(),points[:,0].max(),points[:,1].max()]))
        converted=[dict(class_id=h['box']['class_id'],box_xyxy=[h['box'][k] for k in ('left','top','right','bottom')]) for h in hints]
        summary=dict(status='complete',training_image=Path(report['inspection']).name,real_sam_worker_and_callback=True,
            real_port_button_worker_callback=True,sam_complete=True,initial_alignment_dino_reused=True,
            sam_reference_cache_hit=payloads[0]['reference_sam3']['cache_hit'],sam_inspection_cache_hit=payloads[0]['inspection_sam3']['cache_hit'],
            final_decision=saved['decision'],parent_regions_preserved=True,agent_task_state=load(window.agent_task_path)['state'],
            port_metrics=metric(converted,targets),primary=len(ports['rescue_hints']),supplementary=len(ports['supplementary_hints']),
            student_added=1,independent_accuracy=False,physical_fault_confirmation=False,seconds=round(time.monotonic()-started,2),
            port_evidence=ports['evidence_path'],protected_unchanged=True)
        save(OUT/'acceptance.json',summary);save(OUT/'progress.json',summary);print(json.dumps(summary),flush=True)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error),seconds=round(time.monotonic()-started,2)));raise
    finally:window.close();app.processEvents()
if __name__=='__main__':main()
