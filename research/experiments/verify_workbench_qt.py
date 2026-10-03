"""Render actual default desktop workbench; no model inference or human verdict."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import sys
from pathlib import Path
sys.path.insert(0,'E:/PythonProject10')
sys.path.insert(0,'E:/PythonProject10/tests')
import test_local_review_gui_bridge as fixture
from PyQt5.QtGui import QFontDatabase, QFont

def main():
    fixture.LocalGuiTests.setUpClass()
    app=fixture.LocalGuiTests.app
    app.setStyle('Fusion')
    fid=QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
    if fid>=0: app.setFont(QFont(QFontDatabase.applicationFontFamilies(fid)[0],9))
    test=fixture.LocalGuiTests();test.setUp()
    try:
        w=test.window;w.resize(1600,1100);w.show();app.processEvents()
        out=Path(__file__).resolve().parents[1]/'artifacts'/(sys.argv[1] if len(sys.argv)>1 else 'wiremind_workbench_20261002')
        w.grab().save(str(out/'desktop_window.png'))
        assert not w.agent_review_button.isEnabled()
        assert not w.agent_guidance_button.isEnabled()
        assert not w.agent_local_import_button.isEnabled()
        assert w.run_button.objectName()=='workbenchPrimary'
        w.resize(1366,768);app.processEvents()
        assert w.height()==768, 'Window must fit common laptop screen'
        scroll=w.workbench_controls_scroll
        scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
        app.processEvents()
        assert w.agent_guidance_button.mapTo(scroll.viewport(),w.agent_guidance_button.rect().center()).y()<scroll.viewport().height()
        w.grab().save(str(out/'desktop_small_screen.png'))
        if hasattr(w,'workbench_optional_toggle'):
            assert w.workbench_optional_body.isHidden()
            before=(w.port_hint_button.isEnabled(), w.deepseek_review_button.isEnabled())
            w.workbench_optional_toggle.click();app.processEvents()
            assert not w.workbench_optional_body.isHidden()
            assert (w.port_hint_button.isEnabled(),w.deepseek_review_button.isEnabled())==before
            w.workbench_optional_toggle.click();app.processEvents()
        print('Actual default Qt window rendered; confirmation gates remain disabled.')
    finally: test.doCleanups()

if __name__=='__main__': main()
