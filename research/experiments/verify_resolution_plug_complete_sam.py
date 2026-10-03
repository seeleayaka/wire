"""Gained training image: actual SAM finalization followed by optional Qt worker."""
import copy,json,os,shutil,sys,time
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/resolution_loose_plug_complete_sam_20261003'
OLD=ROOT/'artifacts/student_complete_sam_flow_20261003'
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',YOLO_CONFIG_DIR=str(OUT/'config'))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def main():
    if OUT.exists():raise FileExistsError('Do not duplicate completed or running jobs')
    live=load(ROOT/'artifacts/resolution_loose_plug_live_20261003/report.json');assert live['qualifies_live_diagnostic']
    gui_check=load(ROOT/'artifacts/resolution_loose_plug_gui_20261003/report.json');assert gui_check['status']=='complete'
    SOURCE=Path(live['cases'][0]['initial_report']).parent
    OUT.mkdir();(OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    started=time.monotonic();save(OUT/'progress.json',dict(status='running',phase='verifying_pinned_reference_cache',pid=os.getpid()))
    import psutil,torch
    assert psutil.virtual_memory().available>6*2**30,'Insufficient free memory; no duplicate SAM worker'
    torch.set_num_threads(2)
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation
    from inspection_agent.optional_port_crop_review import sha
    from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint as residual_runtime_fingerprint
    from audit_port_multiscale_acceptance import metric
    import cv2,numpy as np
    app=gui.QApplication.instance() or gui.QApplication([]);window=gui.DINOReview();report=load(SOURCE/'initial_report.json')
    protected={str(p):sha(p) for p in (SOURCE/'initial_report.json',SOURCE/'aligned.jpg',Path(report['inspection']),Path(report['reference']))}
    protected[str(Path(__file__))]=sha(Path(__file__))
    frozen=residual_runtime_fingerprint(REPO)
    for name in ('aligned.jpg','valid_warp_mask.png','check_heatmap.jpg','anomaly_boxes.jpg','dino_anomaly_boxes.jpg'):shutil.copy2(SOURCE/name,OUT/name)
    settings=replace(gui.sam3_wire_fusion.load_settings(),cache_dir=OUT/'sam_cache',threads=2)
    checkpoint_sha=sha(settings.checkpoint)
    old_protocol=load(OLD/'protocol.json');assert old_protocol['sam_checkpoint_sha256']==checkpoint_sha
    assert old_protocol['protected'][str(Path(report['reference']))]==sha(report['reference'])
    ref_key=gui.sam3_wire_fusion._fingerprint(Path(report['reference']))
    old_cache=OLD/'sam_cache/reference_masks'/ref_key
    assert gui.sam3_wire_fusion._report_is_usable(old_cache,Path(report['reference']),settings)
    cache_pins={str(p):sha(p) for p in old_cache.iterdir() if p.is_file()}
    target_cache=settings.cache_dir/'reference_masks'/ref_key
    shutil.copytree(old_cache,target_cache)
    assert all(sha(target_cache/Path(path).name)==digest for path,digest in cache_pins.items())
    save(OUT/'protocol.json',dict(training_selected_image=Path(report['inspection']).name,initial_alignment_dino_reused_sha_verified=True,
        reference_sam_cached_copy=True,reference_cache_source=str(old_cache),reference_cache_pins=cache_pins,
        new_inspection_sam_inference=True,sam_checkpoint_sha256=checkpoint_sha,sam_settings=settings.report(),
        sam_thresholds_unchanged=True,threads_capped=2,protected=protected,runtime_fingerprint=frozen,
        independent_accuracy=False,physical_fault_confirmation=False,automatic_fault_verdict=False))
    save(OUT/'progress.json',dict(status='running',phase='fresh_inspection_sam',pid=os.getpid(),reference_cache_copy_verified=True))
    try:
        window.current_output=OUT;window.reference.setText(report['reference']);window.inspection.setText(report['inspection']);window._write_report(report)
        window._sam3_pending=dict(output=OUT,report=copy.deepcopy(report),dino_regions=copy.deepcopy(report['dino_review_regions']))
        worker=gui.Sam3FusionWorker(Path(report['reference']),OUT/'aligned.jpg',OUT/'valid_warp_mask.png',report['dino_review_regions'],OUT/'check_heatmap.jpg',OUT)
        payloads=[];worker.completed.connect(payloads.append)
        with patch.object(gui.sam3_wire_fusion,'load_settings',return_value=settings):worker.run()
        assert len(payloads)==1;save(OUT/'sam_result.json',payloads[0]);assert payloads[0]['status']=='ok',payloads[0].get('error')
        assert payloads[0]['reference_sam3']['cache_hit'] and not payloads[0]['inspection_sam3']['cache_hit']
        window._finish_sam3_fusion(payloads[0]);app.processEvents();final=load(OUT/'report.json')
        assert final['decision']!='sam3_fusion_running' and final['sam3_fusion']['status']=='ok'
        assert final['image_fingerprints']==report['image_fingerprints'] and final['analysis_check_rois']==report['analysis_check_rois']
        assert window.agent_task_path and window.agent_task_path.is_file();save(OUT/'before_optional_ports.json',final)
        save(OUT/'progress.json',dict(status='running',phase='actual_qt_feature_button',pid=os.getpid()))
        window.port_scene_combo.setCurrentIndex(1)
        for control in (window.rescue_switch,window.rescue_supplement_switch,window.rescue_student_switch,window.rescue_feature_switch):control.setChecked(True)
        window._update_port_controls();assert window.rescue_button.isEnabled();window.rescue_button.click();assert window.rescue_worker is not None
        deadline=time.monotonic()+300
        while window.rescue_worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.02)
        assert window.rescue_worker is None,'Owned port worker timeout'
        app.processEvents();saved=load(OUT/'report.json');ports=saved['independent_port_rescue']
        assert ports['status']=='applied' and ports['feature_residual_policy']['fallback_reason'] is None
        assert ports['resolution_policy']['fallback_reason'] is None
        assert ports['resolution_policy']['added_hints']==1
        assert saved['review_regions']==final['review_regions'] and saved['decision']==final['decision']
        assert len(ports['rescue_hints'])<=5 and len(ports['supplementary_hints'])<=5 and not ports['automatic_fault_verdict']
        assert {p:sha(Path(p)) for p in protected}==protected and residual_runtime_fingerprint(REPO)==frozen
        assert {p:sha(Path(p)) for p in cache_pins}==cache_pins
        targets=[];matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64)
        label=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/labels/train01'/(Path(report['inspection']).stem+'.txt')
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,w,h=map(float,line.split())
            if cls not in (3,4):continue
            l,t,r,b=(cx-w/2)*3648,(cy-h/2)*2736,(cx+w/2)*3648,(cy+h/2)*2736
            points=cv2.perspectiveTransform(np.float32([[[l,t],[r,t],[r,b],[l,b]]]),matrix).reshape(-1,2)
            targets.append(dict(class_id=int(cls)-3,box=[points[:,0].min(),points[:,1].min(),points[:,0].max(),points[:,1].max()]))
        hints=ports['rescue_hints']+ports['supplementary_hints']
        converted=[dict(class_id=h['box']['class_id'],box_xyxy=[h['box'][k] for k in ('left','top','right','bottom')]) for h in hints]
        result=dict(status='complete',sam_complete=True,real_sam_final_callback=True,real_qt_feature_button=True,
            reference_cache_copy=True,new_inspection_sam=True,training_image=Path(report['inspection']).name,
            resolution_added=1,port_metrics=metric(converted,targets),parent_regions_preserved=True,
            final_decision=saved['decision'],agent_state=load(window.agent_task_path)['state'],
            protected_unchanged=True,independent_accuracy=False,physical_fault_confirmation=False,seconds=round(time.monotonic()-started,2))
        assert result['port_metrics']==live['cases'][0]['metrics'], 'Complete-flow source port metrics must match fresh live diagnostic'
        save(OUT/'acceptance.json',result);save(OUT/'progress.json',result);print(json.dumps(result),flush=True)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error),seconds=round(time.monotonic()-started,2)));raise
    finally:window.close();app.processEvents()
if __name__=='__main__':main()
