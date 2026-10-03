"""Fixed replay, real Qt worker/callback and registration equivalence checks."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT=Path(r"E:\PythonProject10")
WORK=Path(__file__).resolve().parents[1]
OUT=WORK / "artifacts/port_crop_gui_20260930"
ACCEPT=WORK / "artifacts/port_crop_acceptance_20260930"
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT / "prototype"))
import assembly_auto_review_dino as gui
import assembly_auto_review_robust_v3 as registration
from inspection_agent.optional_port_crop_review import read_image,sha,REFERENCE_SHA,SCENE
from inspection_agent.port_crop_gui_bridge import run_gui_port_review
import cv2
import numpy as np


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_report(prior,cache,data):
    return {"decision":"possible_difference_manual_review","reference":str(data / "train01/normal_073.JPG"),
            "inspection":str(data / "val01" / prior["image"]),"review_regions":copy.deepcopy(prior["parents"]),
            "existing_port_hints":copy.deepcopy(prior["hints"]),"alignment_quality":{"reliable":True},
            "alignment":{"source_to_reference_homography":prior["actual_homography"]},"local_alignment":[],
            "image_fingerprints":{"source_sha256":cache["source_sha256"],"reference_sha256":REFERENCE_SHA,
                                  "stable_during_visual_analysis":True}}


def main():
    data=ROOT / "data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images"
    old=load(ROOT / "output/port_state_hints_validation_20260929/report.json")
    expected={c["image"]:c for c in load(ACCEPT / "partial_cases.json")}
    application=gui.QApplication.instance() or gui.QApplication([])
    application.setStyle("Fusion")
    # Offscreen Windows Qt does not enumerate system fonts. Load one only here.
    from PyQt5.QtGui import QFontDatabase,QFont
    font_id=QFontDatabase.addApplicationFont(r"C:\Windows\Fonts\msyh.ttc")
    assert font_id>=0
    application.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0],9))
    replay=[]
    for prior in sorted(old["cases"],key=lambda c:c["image"]):
        name=prior["image"]
        cached=load(ACCEPT / "cache" / ("val01_"+Path(name).stem+".json"))
        report=make_report(prior,cached,data)
        original=copy.deepcopy(report)
        result=run_gui_port_review(report,project=ROOT,enabled=True,scene=SCENE,
                                  prediction_provider=lambda image,c=cached:copy.deepcopy(c))
        assert result["status"]=="applied",result.get("fallback_reason")
        for key in ["parents","existing_hints","tile_hints","aligned_predictions","selection_audit"]:
            assert result[key]==expected[name][key],(name,key)
        assert report==original
        replay.append({"image":name,"new_hints":len(result["tile_hints"]),"exact":True})
    assert len(replay)==30 and sum(r["new_hints"] for r in replay)==6
    prior=next(c for c in old["cases"] if c["image"]=="disconnected_002.JPG")
    cache=load(ACCEPT / "cache/val01_disconnected_002.json")
    report=make_report(prior,cache,data)
    gates=[]
    def forbidden(image):
        raise AssertionError("Prediction should not have started")
    for reason,modified,enabled,scene in [
        ("disabled",report,False,SCENE),
        ("visual_geometry_provenance_missing",{**report,"image_fingerprints":{}},True,SCENE),
        ("local_alignment_not_supported",{**report,"local_alignment":[{"dino_alignment_input":"local_ecc_corrected"}]},True,SCENE),
        ("unsupported_scene",report,True,"unknown")]:
        result=run_gui_port_review(modified,project=ROOT,enabled=enabled,scene=scene,prediction_provider=forbidden)
        assert result["tile_hints"]==[] and result["parents"]==report["review_regions"]
        assert result["status"]=="disabled" if reason=="disabled" else result["fallback_reason"]==reason
        gates.append(reason)

    output=OUT / "qt_live_fault"
    output.mkdir(parents=True,exist_ok=True)
    ref=read_image(Path(report["reference"]))
    source=read_image(Path(report["inspection"]))
    aligned=cv2.warpPerspective(source,np.asarray(prior["actual_homography"]),
                               (ref.shape[1],ref.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    gui.adaptive.robust.auto.base.write_image(output / "aligned.jpg",aligned)
    window=gui.DINOReview()
    try:
        assert not window.port_hint_switch.isChecked() and not window.port_hint_button.isEnabled()
        window.reference.setText(report["reference"])
        window.inspection.setText(report["inspection"])
        window.current_output=output
        window._write_report(report)
        window.set_decision("uncertain","原视觉报告回放：待人工复核","固定父框回放；端口提示本次实际推理。")
        window._create_agent_task(report)
        assert not window.port_hint_button.isEnabled()
        window.port_hint_switch.setChecked(True)
        window.port_scene_combo.setCurrentIndex(1)
        assert window.port_hint_button.isEnabled()
        window.inspection.setText("different_source.JPG")
        window._update_port_controls()
        assert not window.port_hint_button.isEnabled()
        window.inspection.setText(report["inspection"])
        window._update_port_controls()
        assert window.port_hint_button.isEnabled()
        worker_ref=[]
        window.start_port_crop_review()
        worker=window.port_worker
        worker_ref.append(worker)
        assert worker is not None and not window.port_hint_button.isEnabled()
        finished=[]
        worker.finished.connect(lambda:finished.append(True))
        ticks=0;deadline=time.monotonic()+90
        while (window.port_worker is not None or not finished) and time.monotonic()<deadline:
            application.processEvents();time.sleep(.015);ticks+=1
        assert finished and window.port_worker is None,"Qt worker timed out"
        saved=load(output / "report.json")
        live=load(output / "port_crop_review.json")
        assert live["status"]=="applied" and len(live["tile_hints"])==1
        assert live["tile_hints"]==expected["disconnected_002.JPG"]["tile_hints"]
        assert saved["review_regions"]==report["review_regions"]
        assert window.port_hint_view.item is not None
        assert window.port_comparison_panel is not None and window.port_hint_button.isEnabled()
        window.resize(1500,1000);window.show();application.processEvents()
        assert window.grab().save(str(OUT / "qt_port_hint.png"))
        window.tabs.setCurrentWidget(window.port_comparison_panel);application.processEvents()
        assert window.grab().save(str(OUT / "qt_port_comparison.png"))
        window.port_hint_switch.setChecked(False)
        assert window.port_hint_view.item is None and not window.port_hint_button.isEnabled()
        assert window.port_hint_view.scene.items()==[]
        # Obsolete results must not paint a later task.
        window.current_output=OUT / "new_task"
        window._finish_port_crop_review({"output":str(output),"result":live})
        assert window.port_hint_view.item is None
    finally:
        window.close();application.processEvents()

    # Compare registration before/after with the same fixed seed and source.
    current=registration.automatic_homography
    original_auto=registration.auto.automatic_affine
    namespace={"__name__":"original_registration_snapshot","__file__":str(OUT / "before/assembly_auto_review_robust_v3.py")}
    try:
        exec(compile((OUT / "before/assembly_auto_review_robust_v3.py").read_text(encoding="utf-8"),namespace["__file__"],"exec"),namespace)
        cv2.setRNGSeed(0);before_image,before_report=namespace["automatic_homography"](ref,source)
        cv2.setRNGSeed(0);after_image,after_report=current(ref,source)
    finally:
        registration.auto.automatic_affine=original_auto
    matrix=after_report.pop("source_to_reference_homography")
    assert np.array_equal(before_image,after_image) and before_report==after_report
    assert matrix==prior["actual_homography"]
    result={"status":"verified","fixed_gui_bridge_replays":30,"new_hints":6,"bridge_gates":gates,
            "qt_real_inference_hints":1,"qt_event_loop_iterations":ticks,
            "original_regions_unchanged":True,"default_off_verified":True,"off_clears_display":True,
            "stale_result_ignored":True,"registration_pixels_and_old_fields_identical":True,
            "changed_input_prevents_old_geometry_reuse":True,
            "homography_matches_frozen_validation":True,
            "scope":"Cached fixed-parent replay plus real Qt port worker, not fresh complete DINO/SAM acceptance.",
            "fingerprints":{str(ROOT / p):sha(ROOT / p) for p in
                ["prototype/assembly_auto_review_dino.py","prototype/assembly_auto_review_robust_v3.py",
                 "inspection_agent/port_crop_gui_bridge.py"]},"cases":replay}
    (OUT / "verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("cases","fingerprints")},ensure_ascii=False))


if __name__=="__main__":
    main()
