"""Actual Qt-button smoke check with synthetic operator-shaped fixture, not field validation."""
import json
import os
from pathlib import Path
import sys
os.environ['QT_QPA_PLATFORM']='offscreen'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/local_review_gui_20261002'
sys.path.insert(0,'E:/PythonProject10');sys.path.insert(0,'E:/PythonProject10/tests')
import test_local_review_gui_bridge as test_module
from PyQt5.QtGui import QFontDatabase,QFont
def main():
    OUT.mkdir(parents=True,exist_ok=True);test_module.LocalGuiTests.setUpClass()
    app=test_module.LocalGuiTests.app;app.setStyle('Fusion')
    fid=QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
    if fid>=0:app.setFont(QFont(QFontDatabase.applicationFontFamilies(fid)[0],9))
    test=test_module.LocalGuiTests();test.setUp()
    try:
        test.operator_fixture();test.ready();test.click();w=test.window
        w.resize(1400,1000);w.show();app.processEvents()
        w.result_card.grab().save(str(OUT/'qt_result_card.png'))
        w.agent_local_summary.grab().save(str(OUT/'qt_import_summary.png'))
        report=test_module.InspectionTask.load(test.fixture.tp).to_report()
        report['verification_context']=dict(synthetic_fixture=True,operator_mode_shape_only=True,
            actual_operator_confirmation=False,new_visual_inference=False)
        (OUT/'synthetic_task_after_gui_import.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        checks=dict(actual_button_clicked=True,state=report['state'],summary=w.agent_local_summary.text(),
            modification_guidance_enabled=w.agent_guidance_button.isEnabled(),human_conclusions=report['human_conclusions'],
            synthetic_fixture=True,actual_operator_confirmation=False)
        assert checks['state']=='awaiting_human_review' and not checks['modification_guidance_enabled'] and not checks['human_conclusions']
        (OUT/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Qt button, stored task and summary checked; synthetic fixture only.')
    finally:test.doCleanups()
if __name__=='__main__':main()
