"""Fresh inspection SAM and actual Qt median-geometry enhancement acceptance."""
import copy,json,os,shutil,sys,time
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/paired_median_complete_sam_20261003'
OLD=ROOT/'artifacts/student_complete_sam_flow_20261003'
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',YOLO_CONFIG_DIR=str(OUT/'config'))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');tmp.replace(p)


def main():
    if OUT.exists():raise FileExistsError('Preserve complete-flow jobs')
    live=load(ROOT/'artifacts/paired_median_live_20261003/report.json');assert live['qualifies_live_diagnostic']
    positive=next(row for row in live['cases'] if row['stage']=='train' and row['gained'])
    SOURCE=Path(positive['initial_report']).parent
    OUT.mkdir();(OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    started=time.monotonic();save(OUT/'progress.json',dict(status='running',phase='reference_cache_and_default_off',pid=os.getpid()))
    import psutil,torch
    assert psutil.virtual_memory().available>6*2**30,'Insufficient free memory; no duplicate SAM'
    torch.set_num_threads(2)
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation
    from inspection_agent.optional_port_crop_review import sha,SCENE
    from inspection_agent.paired_median_geometry import median_runtime_fingerprint as paired_runtime_fingerprint,POLICY_ID,HEAD_SHA
    from inspection_agent.port_rescue_gui import snapshot,payload_is_current,finish_port_rescue
    from paired_graph_hint_link import accepted_native_rows
    from current_port_baseline_audit import read_targets,BASE
    from audit_port_multiscale_acceptance import metric,matches
    import numpy as np
    app=gui.QApplication.instance() or gui.QApplication([]);window=gui.DINOReview();report=load(SOURCE/'initial_report.json')
    controls=(window.rescue_switch,window.rescue_supplement_switch,window.rescue_student_switch,window.rescue_feature_switch)
    assert not any(c.isChecked() for c in controls)
    protected={str(p):sha(p) for p in (SOURCE/'initial_report.json',SOURCE/'aligned.jpg',Path(report['inspection']),Path(report['reference']),Path(positive['evidence']),Path(__file__))}
    frozen=paired_runtime_fingerprint(REPO)
    for name in ('aligned.jpg','valid_warp_mask.png','check_heatmap.jpg','anomaly_boxes.jpg','dino_anomaly_boxes.jpg'):shutil.copy2(SOURCE/name,OUT/name)
    settings=replace(gui.sam3_wire_fusion.load_settings(),cache_dir=OUT/'sam_cache',threads=2)
    checkpoint_sha=sha(settings.checkpoint);old_protocol=load(OLD/'protocol.json');assert old_protocol['sam_checkpoint_sha256']==checkpoint_sha
    assert old_protocol['protected'][str(Path(report['reference']))]==sha(report['reference'])
    ref_key=gui.sam3_wire_fusion._fingerprint(Path(report['reference']));old_cache=OLD/'sam_cache/reference_masks'/ref_key
    assert gui.sam3_wire_fusion._report_is_usable(old_cache,Path(report['reference']),settings)
    cache_pins={str(p):sha(p) for p in old_cache.iterdir() if p.is_file()};target_cache=settings.cache_dir/'reference_masks'/ref_key
    shutil.copytree(old_cache,target_cache);assert all(sha(target_cache/Path(p).name)==digest for p,digest in cache_pins.items())
    save(OUT/'protocol.json',dict(selected_first_training_gain=positive,head_sha256=HEAD_SHA,runtime_fingerprint=frozen,
        protected=protected,reference_sam_cache_copy_verified=True,reference_cache_pins=cache_pins,
        sam_checkpoint_sha256=checkpoint_sha,sam_settings=settings.report(),new_inspection_sam=True,
        default_off=True,manual_review_only=True,field_accuracy=False))
    try:
        window.current_output=OUT;window.reference.setText(report['reference']);window.inspection.setText(report['inspection']);window._write_report(report)
        window.port_scene_combo.setCurrentIndex(1)
        for control in controls:control.setChecked(True)
        window._update_port_controls();assert not window.rescue_button.isEnabled(),'SAM pending must not be relabeled'
        for control in controls:control.setChecked(False)
        window._sam3_pending=dict(output=OUT,report=copy.deepcopy(report),dino_regions=copy.deepcopy(report['dino_review_regions']))
        save(OUT/'progress.json',dict(status='running',phase='fresh_inspection_sam',pid=os.getpid()))
        worker=gui.Sam3FusionWorker(Path(report['reference']),OUT/'aligned.jpg',OUT/'valid_warp_mask.png',report['dino_review_regions'],OUT/'check_heatmap.jpg',OUT)
        payloads=[];worker.completed.connect(payloads.append)
        with patch.object(gui.sam3_wire_fusion,'load_settings',return_value=settings):worker.run()
        assert len(payloads)==1;save(OUT/'sam_result.json',payloads[0]);assert payloads[0]['status']=='ok',payloads[0].get('error')
        assert payloads[0]['reference_sam3']['cache_hit'] and not payloads[0]['inspection_sam3']['cache_hit']
        window._finish_sam3_fusion(payloads[0]);app.processEvents();final=load(OUT/'report.json')
        assert final['decision']!='sam3_fusion_running' and final['sam3_fusion']['status']=='ok'
        assert final['image_fingerprints']==report['image_fingerprints'] and final['analysis_check_rois']==report['analysis_check_rois']
        assert window.agent_task_path and window.agent_task_path.is_file();save(OUT/'before_optional_ports.json',final)
        save(OUT/'progress.json',dict(status='running',phase='actual_qt_enhancement_button',pid=os.getpid()))
        for control in controls:control.setChecked(True)
        window._update_port_controls();assert window.rescue_button.isEnabled() and snapshot(window)['policy']==POLICY_ID
        before=sha(OUT/'report.json');finish_port_rescue(window,dict(token='stale',snapshot=snapshot(window),result={},evidence='unused'));assert sha(OUT/'report.json')==before
        window.rescue_button.click();assert window.rescue_worker is not None
        deadline=time.monotonic()+420
        while window.rescue_worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.02)
        assert window.rescue_worker is None,'Owned Qt worker timeout'
        app.processEvents();saved=load(OUT/'report.json');ports=saved['independent_port_rescue'];evidence=load(Path(ports['evidence_path'])/'evidence.json')['result']
        assert ports['status']=='applied' and ports['median_geometry_policy']['fallback_reason'] is None
        assert ports['median_geometry_policy']['added_hints']==positive['accepted']
        assert ports['median_geometry_policy']['head_sha256']==HEAD_SHA
        old=load(SOURCE/'current_paired_ports.json');diagnostic=load(Path(positive['evidence']))
        assert ports['rescue_hints']==old['rescue_hints']
        assert ports['supplementary_hints'][:len(old['supplementary_hints'])]==old['supplementary_hints']
        normalize=lambda rows:[{k:v for k,v in h.items() if k not in ('paired_geometry_policy_id','median_geometry_policy_id')} for h in rows]
        assert normalize(ports['supplementary_hints'])==normalize(diagnostic['supplementary_hints'])
        assert saved['review_regions']==final['review_regions'] and saved['decision']==final['decision']
        assert len(ports['rescue_hints'])<=5 and len(ports['supplementary_hints'])<=5 and not ports['automatic_fault_verdict']
        window._rescue_token='head_freshness';bound=snapshot(window);payload=dict(token='head_freshness',snapshot=bound)
        assert payload_is_current(window,payload)
        changed=copy.deepcopy(bound['feature_runtime']);changed['head']='changed'
        saved_sha=sha(OUT/'report.json')
        with patch('inspection_agent.port_rescue_gui.residual_runtime_fingerprint',return_value=changed):
            assert not payload_is_current(window,payload)
            finish_port_rescue(window,{**payload,'result':{},'evidence':'unused'})
        assert saved_sha==sha(OUT/'report.json')
        geometry_changed=copy.deepcopy(bound['feature_runtime']);geometry_changed['geometry']='changed'
        with patch('inspection_agent.port_rescue_gui.residual_runtime_fingerprint',return_value=geometry_changed):
            assert not payload_is_current(window,payload)
            finish_port_rescue(window,{**payload,'result':{},'evidence':'unused'})
        assert saved_sha==sha(OUT/'report.json')
        window.rescue_feature_switch.setChecked(False);assert not payload_is_current(window,payload)
        assert {p:sha(Path(p)) for p in protected}==protected and paired_runtime_fingerprint(REPO)==frozen
        assert {p:sha(Path(p)) for p in cache_pins}==cache_pins
        extra=ports['supplementary_hints'][len(old['supplementary_hints']):]
        accepted=accepted_native_rows(evidence['median_geometry_evidence']['native']['paired_semantic_additions'],extra,
            np.asarray(report['alignment']['source_to_reference_homography']),[2736,3648],[2736,3648])
        entry=next(row for row in load(BASE/'train/report.json')['cases'] if row['image']==positive['image'])
        label_pins={};targets=read_targets('train',positive['image'],[2736,3648],entry['label_sha256'],label_pins)
        current=evidence['median_geometry_evidence']['native_current']['all_predictions'];result_metrics=metric(current+accepted,targets)
        assert result_metrics==positive['trial'] and not (matches(current,targets)[0]-matches(current+accepted,targets)[0])
        result=dict(status='complete',sam_complete=True,new_inspection_sam=True,reference_sam_cache_copy=True,
            actual_qt_button_worker_callback=True,default_off=True,sam_pending_blocked=True,head_drift_stale_rejected=True,geometry_drift_stale_rejected=True,
            original_cues_preserved=True,median_added=len(accepted),source_port_metrics=result_metrics,
            final_decision=saved['decision'],agent_state=load(window.agent_task_path)['state'],
            protected_unchanged=True,field_accuracy=False,physical_fault_confirmation=False,seconds=round(time.monotonic()-started,2))
        save(OUT/'acceptance.json',result);save(OUT/'progress.json',result);print(json.dumps(result),flush=True)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error),seconds=round(time.monotonic()-started,2)));raise
    finally:window.close();app.processEvents()

if __name__=='__main__':main()
