"""Exercise script-mode entry in a fresh interpreter, not a cached module import."""
import os,subprocess,sys,unittest
from pathlib import Path

class LauncherStartupTests(unittest.TestCase):
    def test_script_entry_constructs_real_window(self):
        script=Path(__file__).resolve().with_name('launch_llm_recheck_window_20261008.py')
        root=Path(os.environ.get('WIREMIND_PROJECT_ROOT','E:/PythonProject10'))
        child="""
import os,sys,runpy
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1]).parent))
root=Path(os.environ['WIREMIND_PROJECT_ROOT'])
sys.path[:0]=[str(root),str(root/'models/dinov2'),str(root/'prototype')]
import assembly_auto_review_dino_v2 as entry
import assembly_auto_review_dino as gui
def smoke():
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    app=QApplication([])
    window=gui.DINOReview()
    assert type(window).__name__=='PlannedWindow'
    window.show();app.processEvents()
    assert window.isVisible() and window.cloud_body.isHidden()
    window.cloud_switch.setChecked(True);app.processEvents()
    assert window.cloud_body.isVisible() and window.thinking_mode.count()==4
    QTimer.singleShot(100,app.quit)
    app.exec_();window.close()
    print('PRODUCTION_SCRIPT_STARTUP_OK')
entry.main=smoke
runpy.run_path(sys.argv[1],run_name='__main__')
"""
        env={**os.environ,'QT_QPA_PLATFORM':'offscreen','PYTHONDONTWRITEBYTECODE':'1','WIREMIND_PROJECT_ROOT':str(root)}
        result=subprocess.run([sys.executable,'-B','-c',child,str(script)],env=env,cwd=str(script.parent.parent),capture_output=True,text=True,timeout=45)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('PRODUCTION_SCRIPT_STARTUP_OK',result.stdout)
if __name__=='__main__':unittest.main()
