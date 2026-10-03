"""Actual desktop worker and result callback on a copied, real SAM report."""
import os
import sys
import json
import shutil
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['QT_QPA_PLATFORM']='offscreen'
REPO=Path('E:/PythonProject10');WORK=Path(__file__).resolve().parents[1]
OUT=WORK/'artifacts/port_bridge_fix_20261002/qt'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'prototype'))
import torch
import assembly_auto_review_dino_v2 as entry
gui=entry.implementation
from inspection_agent.optional_port_crop_review import SCENE
source=WORK/'artifacts/main_scenario_live_20261002/disconnected_005/20261002_131828'
report=json.loads((source/'report.json').read_text(encoding='utf-8'))
shutil.copy2(source/'aligned.jpg',OUT/'aligned.jpg')
app=gui.QApplication.instance() or gui.QApplication([])
window=gui.DINOReview()
try:
    assert not window.port_hint_switch.isChecked()
    window.current_output=OUT
    window.reference.setText(report['reference']);window.inspection.setText(report['inspection'])
    window._write_report(report)
    window.port_hint_switch.setChecked(True);window.port_scene_combo.setCurrentIndex(1)
    assert window.port_hint_button.isEnabled()
    worker=gui.PortCropReviewWorker(report,OUT,SCENE)
    payload=[];worker.completed.connect(payload.append);worker.run()
    assert len(payload)==1
    result=payload[0]['result'];assert result['status']=='applied',result['fallback_reason']
    window._finish_port_crop_review(payload[0]);app.processEvents()
    saved=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
    assert saved['review_regions']==report['review_regions']
    assert window.port_hint_view.item is not None
    assert window.port_hint_button.isEnabled()
    assert not window.agent_guidance_button.isEnabled()
    (OUT/'verification.json').write_text(json.dumps(dict(actual_worker_and_callback=True,
        input_visual_report='real_SAM_fusion_saved_report',new_port_inference=True,
        new_visual_inference=False,new_hints=len(result['tile_hints']),
        baseline_candidates_unchanged=True,repair_guidance_disabled=True),indent=2)+'\n',encoding='utf-8')
    print('QT WORKER/CALLBACK VERIFIED')
finally:
    window.close();app.processEvents()
