"""Full fresh inspection-SAM and actual candidate Qt button in a disposable fixture."""
import copy
import ast
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
REPO=Path(os.environ.get('WIRE_NATIVE_PROJECT',str(ROOT/'staging/native_pose_acceptance_project')))
OUT=ROOT/'artifacts'/os.environ.get('WIRE_NATIVE_COMPLETE_OUT','native_portable_complete_sam_20261004')
LIVE=ROOT/'artifacts/paired_pose_native_live_recovered_20261004/report.json'
OLD=ROOT/'artifacts/student_complete_sam_flow_20261003'
REUSE_SAM=os.environ.get('WIRE_REUSE_VERIFIED_SAM')
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',
                  YOLO_CONFIG_DIR=str(OUT/'config'),PYTHONPROJECT10_DINO_CACHE=str(OUT/'dino_cache'))
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def main():
    if OUT.exists():raise FileExistsError('Preserve full integration evidence')
    live=load(LIVE);assert live['qualifies']
    positive=next(row for row in live['cases'] if row['stage']=='train' and row['gained'])
    source=Path(positive['report']).parent
    OUT.mkdir();(OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    started=time.monotonic();save(OUT/'progress.json',dict(status='running',phase='default_off_and_reference_cache',pid=os.getpid()))
    import psutil
    assert psutil.virtual_memory().available>6*2**30,'No duplicate SAM or memory-intensive audit'
    import torch
    torch.set_num_threads(2)
    import assembly_auto_review_dino_v2 as entry
    gui=entry.implementation
    from inspection_agent import paired_native_pose as native
    from inspection_agent.port_rescue_gui import snapshot,payload_is_current,finish_port_rescue
    from inspection_agent.optional_port_crop_review import SCENE
    assert Path(native.__file__).is_relative_to(REPO),'Wrong codebase imported'
    app=gui.QApplication.instance() or gui.QApplication([]);window=gui.DINOReview();report=load(positive['report'])
    controls=(window.rescue_switch,window.rescue_supplement_switch,window.rescue_student_switch,window.rescue_feature_switch)
    assert not any(c.isChecked() for c in controls)
    frozen=native.native_pose_runtime_fingerprint(REPO);native.validate_release(load(REPO/'config'/native.MANIFEST_NAME),frozen)
    protected={str(p):sha(p) for p in (Path(positive['report']),source/'aligned.jpg',Path(report['inspection']),Path(report['reference']),Path(positive['evidence']),Path(__file__))}
    for name in ('aligned.jpg','valid_warp_mask.png','check_heatmap.jpg','anomaly_boxes.jpg','dino_anomaly_boxes.jpg'):shutil.copy2(source/name,OUT/name)
    # Dependencies stay in the existing isolated SAM environment; inputs and model
    # use the fixture's byte-identical checkpoint and licensed source snapshot.
    settings=replace(gui.sam3_wire_fusion.load_settings(),cache_dir=OUT/'sam_cache',threads=2,
                     python=Path('E:/PythonProject10/runtime/sam3/.venv/Scripts/python.exe'))
    protocol=load(OLD/'protocol.json');assert protocol['sam_checkpoint_sha256']==sha(settings.checkpoint)
    key=gui.sam3_wire_fusion._fingerprint(Path(report['reference']));old_cache=OLD/'sam_cache/reference_masks'/key
    assert gui.sam3_wire_fusion._report_is_usable(old_cache,Path(report['reference']),settings)
    cache_pins={str(p):sha(p) for p in old_cache.iterdir() if p.is_file()}
    target_cache=settings.cache_dir/'reference_masks'/key;shutil.copytree(old_cache,target_cache)
    assert all(sha(target_cache/Path(p).name)==digest for p,digest in cache_pins.items())
    save(OUT/'protocol.json',dict(positive=positive,pins=protected,runtime_fingerprint=frozen,
         checkpoint_sha256=sha(settings.checkpoint),reference_cache_copy_sha_verified=True,
         selected_training_integration_only=True,actual_qt_fixture_code=True,field_accuracy=False))
    try:
        window.current_output=OUT;window.reference.setText(report['reference']);window.inspection.setText(report['inspection']);window._write_report(report)
        window.port_scene_combo.setCurrentIndex(1)
        for control in controls:control.setChecked(True)
        window._update_port_controls();assert not window.rescue_button.isEnabled(),'SAM pending must remain blocked'
        for control in controls:control.setChecked(False)
        window._sam3_pending=dict(output=OUT,report=copy.deepcopy(report),dino_regions=copy.deepcopy(report['dino_review_regions']))
        save(OUT/'progress.json',dict(status='running',phase='fresh_inspection_sam',pid=os.getpid()))
        worker=gui.Sam3FusionWorker(Path(report['reference']),OUT/'aligned.jpg',OUT/'valid_warp_mask.png',report['dino_review_regions'],OUT/'check_heatmap.jpg',OUT)
        payloads=[];worker.completed.connect(payloads.append)
        if REUSE_SAM:
            prior=ROOT/'artifacts'/REUSE_SAM
            prior_protocol=load(prior/'protocol.json')
            if REPO==Path('E:/PythonProject10'):
                # Promotion changes only the accepted manifest binding, not inference.
                fixture=ROOT/'staging/native_pose_acceptance_project'
                old=prior_protocol['runtime_fingerprint']
                assert all(old[k]==frozen[k] for k in ('accepted','head','encoder','geometry'))
                assert sha(fixture/'inspection_agent/paired_native_pose.py')==old['backend']
                assert sha(fixture/'config'/native.MANIFEST_NAME)==old['manifest']
                def algorithm(path):
                    tree=ast.parse(path.read_text(encoding='utf-8'))
                    for node in tree.body:
                        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MANIFEST_SHA' for t in node.targets):
                            node.value=ast.Constant('verified_promotion_binding')
                    return ast.dump(tree,include_attributes=False)
                assert algorithm(fixture/'inspection_agent/paired_native_pose.py')==algorithm(REPO/'inspection_agent/paired_native_pose.py')
                for relative in ('inspection_agent/port_rescue_gui.py','prototype/assembly_auto_review_dino_v2.py','prototype/assembly_auto_review_dino.py'):
                    assert sha(fixture/relative)==sha(REPO/relative)
            else:
                assert prior_protocol['runtime_fingerprint']==frozen
            assert prior_protocol['checkpoint_sha256']==sha(settings.checkpoint)
            assert sha(prior/'aligned.jpg')==sha(OUT/'aligned.jpg')
            assert all(sha(p)==value for p,value in prior_protocol['pins'].items()
                       if p!=str(Path(__file__)))
            payloads= [load(prior/'sam_result.json')]
            save(OUT/'verified_fresh_sam_reuse.json',dict(prior=str(prior),
                 payload_sha256=sha(prior/'sam_result.json'),aligned_sha256=sha(OUT/'aligned.jpg'),
                 current_run_recomputed_sam=False,prior_fresh_inspection_sam=True))
        else:
            with patch.object(gui.sam3_wire_fusion,'load_settings',return_value=settings):worker.run()
        assert len(payloads)==1,payloads;save(OUT/'sam_result.json',payloads[0]);assert payloads[0]['status']=='ok',payloads[0].get('error')
        assert payloads[0]['reference_sam3']['cache_hit'] and not payloads[0]['inspection_sam3']['cache_hit']
        window._finish_sam3_fusion(payloads[0]);app.processEvents();final=load(OUT/'report.json')
        assert final['decision']!='sam3_fusion_running' and final['sam3_fusion']['status']=='ok'
        assert final['image_fingerprints']==report['image_fingerprints'] and final['analysis_check_rois']==report['analysis_check_rois']
        assert window.agent_task_path and window.agent_task_path.is_file();save(OUT/'before_optional_ports.json',final)
        for control in controls:control.setChecked(True)
        window._update_port_controls();assert window.rescue_button.isEnabled() and snapshot(window)['policy']==native.POLICY_ID
        save(OUT/'progress.json',dict(status='running',phase='actual_qt_native_button',pid=os.getpid()))
        before=sha(OUT/'report.json');finish_port_rescue(window,dict(token='stale',snapshot=snapshot(window),result={},evidence='unused'));assert sha(OUT/'report.json')==before
        window.rescue_button.click();assert window.rescue_worker is not None
        deadline=time.monotonic()+480
        while window.rescue_worker is not None and time.monotonic()<deadline:app.processEvents();time.sleep(.02)
        assert window.rescue_worker is None,'Candidate Qt worker timeout'
        app.processEvents();saved=load(OUT/'report.json');ports=saved['independent_port_rescue']
        evidence=load(Path(ports['evidence_path'])/'evidence.json')['result'];policy=ports['native_pose_policy']
        assert ports['status']=='applied' and policy['fallback_reason'] is None,policy
        assert policy['head_sha256']==native.HEAD_SHA and policy['added_hints']==positive['accepted']
        original=load(source/'accepted_median_ports.json');actual=load(positive['evidence'])
        assert ports['rescue_hints']==original['rescue_hints']
        assert ports['supplementary_hints'][:len(original['supplementary_hints'])]==original['supplementary_hints']
        new=ports['supplementary_hints'][len(original['supplementary_hints']):]
        expected=actual['supplementary_hints'][len(original['supplementary_hints']):]
        suffix='Reference non-detection does not prove physical fault or continuity.'
        assert all(h.get('warning','').endswith(suffix) for h in new+expected)
        normalize=lambda rows:[{k:v for k,v in row.items() if k not in ('native_pose_policy_id','median_geometry_policy_id','warning')} for row in rows]
        assert normalize(new)==normalize(expected),'Portable final cue differs from original actual adapter'
        assert saved['review_regions']==final['review_regions'] and saved['decision']==final['decision']
        assert len(ports['rescue_hints'])<=5 and len(ports['supplementary_hints'])<=5 and not ports['automatic_fault_verdict']
        window._rescue_token='native_freshness';binding=snapshot(window);payload=dict(token='native_freshness',snapshot=binding)
        assert payload_is_current(window,payload);saved_sha=sha(OUT/'report.json')
        for key in ('head','geometry','manifest'):
            changed=copy.deepcopy(binding['feature_runtime']);changed[key]='changed'
            with patch('inspection_agent.port_rescue_gui.residual_runtime_fingerprint',return_value=changed):
                assert not payload_is_current(window,payload);finish_port_rescue(window,{**payload,'result':{},'evidence':'unused'})
            assert saved_sha==sha(OUT/'report.json')
        window.rescue_feature_switch.setChecked(False);assert not payload_is_current(window,payload)
        assert {p:sha(p) for p in protected}==protected and native.native_pose_runtime_fingerprint(REPO)==frozen
        result=dict(status='complete',sam_complete=True,new_inspection_sam=True,reference_cache_copy_sha_verified=True,
             actual_qt_button_worker_callback=True,default_off=True,sam_pending_blocked=True,
             head_geometry_manifest_stale_rejected=True,original_cues_preserved=True,new_added=len(new),
             matched_development_metrics=positive['trial'],agent_state=load(window.agent_task_path)['state'],
             field_accuracy=False,fixture_only=REPO!=Path('E:/PythonProject10'),no_E_deployment=REPO!=Path('E:/PythonProject10'),
             prior_fresh_SAM_reused_SHA_verified=bool(REUSE_SAM),current_run_new_SAM=not bool(REUSE_SAM),
             intentional_warning_title_change_only=True,seconds=round(time.monotonic()-started,2))
        save(OUT/'acceptance.json',result);save(OUT/'progress.json',result);print(json.dumps(result),flush=True)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error),seconds=round(time.monotonic()-started,2)));raise
    finally:window.close();app.processEvents()

if __name__=='__main__':main()
