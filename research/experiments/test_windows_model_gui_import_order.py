"""Regression seam: fresh process mirrors the real model/Qt bootstrap order."""
import os
import subprocess
import sys
import unittest


class WindowsImportOrderTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform=='win32','Windows DLL initialization contract')
    def test_torch_before_qt_in_fresh_process(self):
        env=dict(os.environ,QT_QPA_PLATFORM='offscreen')
        result=subprocess.run([sys.executable,'-B','-c',
            "import torch; from PyQt5.QtWidgets import QApplication; app=QApplication([]); print('ordered_boot_ok')"],
            env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('ordered_boot_ok',result.stdout)


if __name__=='__main__':unittest.main()
