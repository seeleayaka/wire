"""Render saved real Qt worker output at the final viewport size, no inference."""
import os
import sys
from pathlib import Path
import json

os.environ["QT_QPA_PLATFORM"]="offscreen"
ROOT=Path(r"E:\PythonProject10")
OUT=Path(__file__).resolve().parents[1] / "artifacts/port_crop_gui_20260930"
sys.path.insert(0,str(ROOT / "prototype"))
sys.path.insert(0,str(ROOT))
import assembly_auto_review_dino as gui
from PyQt5.QtGui import QFont,QFontDatabase

application=gui.QApplication([])
font=QFontDatabase.addApplicationFont(r"C:\Windows\Fonts\msyh.ttc")
assert font>=0
application.setFont(QFont(QFontDatabase.applicationFontFamilies(font)[0],9))
application.setStyle("Fusion")
report=json.loads((OUT / "qt_live_fault/report.json").read_text(encoding="utf-8"))
result=json.loads((OUT / "qt_live_fault/port_crop_review.json").read_text(encoding="utf-8"))
window=gui.DINOReview()
try:
    window.reference.setText(report["reference"])
    window.inspection.setText(report["inspection"])
    window.current_output=OUT / "qt_live_fault"
    window._write_report(report)
    window.agent_task_path=window.current_output / "agent_task.json"
    window._set_agent_state("awaiting_human_review",window.agent_task_path)
    window.set_decision("uncertain","原视觉报告回放：待人工复核","显示本次已保存的真实端口推理结果。")
    window.port_hint_switch.setChecked(True)
    window.port_scene_combo.setCurrentIndex(1)
    window.resize(1500,1000)
    window.show();application.processEvents()
    window._finish_port_crop_review({"output":str(window.current_output),"result":result})
    application.processEvents();window.port_hint_view.fit_image()
    assert window.grab().save(str(OUT / "qt_port_hint.png"))
    window.tabs.setCurrentWidget(window.port_comparison_panel);application.processEvents()
    window.port_comparison_panel.reference_view.fit_image()
    window.port_comparison_panel.inspection_view.fit_image()
    assert window.grab().save(str(OUT / "qt_port_comparison.png"))
    print("Rendered saved actual worker result; no inference rerun")
finally:
    window.close()
