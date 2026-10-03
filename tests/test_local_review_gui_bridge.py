import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'prototype'))
import assembly_auto_review_dino as gui
import test_local_evidence_bridge as fixture_module
from inspection_agent import InspectionTask

class LocalGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=gui.QApplication.instance() or gui.QApplication([])
    def setUp(self):
        self.fixture=fixture_module.LocalEvidenceTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture
        manifest=f.root/'files.json';manifest.write_text(json.dumps({c:{k:str(v) for k,v in paths.items()} for c,paths in f.files.items()}),encoding='utf-8')
        self.packet=f.root/'packet.json';self.packet.write_text(json.dumps(dict(schema_version=1,bundle=str(f.bp),review=str(f.rp),evidence_files=str(manifest),plan=None)),encoding='utf-8')
        self.window=gui.DINOReview();self.addCleanup(self.window.close)
        self.window.reference.setText(str(f.image));self.window.inspection.setText(str(f.image));self.window.agent_task_path=f.tp
    def ready(self):self.window._set_agent_state('awaiting_human_review',self.fixture.tp)
    def operator_fixture(self):
        self.fixture.review['review_mode']='operator_review';self.fixture.dump(self.fixture.rp,self.fixture.review)
    def click(self):
        with patch.object(gui.QFileDialog,'getOpenFileName',return_value=(str(self.packet),'JSON')):self.window.agent_local_import_button.click()
    def test_initial_controls_disabled(self):self.assertFalse(self.window.agent_local_import_button.isEnabled())
    def test_actual_button_import_and_old_controls(self):
        self.operator_fixture();self.ready();self.click();task=InspectionTask.load(self.fixture.tp)
        self.assertEqual(task.state,'awaiting_human_review');self.assertEqual(len(task.to_report()['local_evidence_reviews']),1)
        self.assertEqual(task.to_report()['human_conclusions'],[]);self.assertIn('支持1',self.window.agent_local_summary.text())
        self.assertFalse(self.window.agent_guidance_button.isEnabled());self.assertTrue(self.window.agent_review_button.isEnabled())
    def test_fixture_default_reject_and_no_write(self):
        self.ready();before=self.fixture.tp.read_bytes();self.click()
        self.assertEqual(before,self.fixture.tp.read_bytes());self.assertIn('未导入',self.window.agent_local_summary.text())
    def test_wrong_current_photo_no_write(self):
        self.operator_fixture();self.ready();before=self.fixture.tp.read_bytes();self.window.inspection.setText(str(self.fixture.root/'other.png'));self.click()
        self.assertEqual(before,self.fixture.tp.read_bytes());self.assertIn('当前图片',self.window.agent_local_summary.text())
    def test_changed_task_during_picker_rejected(self):
        self.operator_fixture();self.ready()
        def mutate(*args):
            self.fixture.tp.write_bytes(self.fixture.tp.read_bytes()+b' ')
            return str(self.packet),'JSON'
        with patch.object(gui.QFileDialog,'getOpenFileName',side_effect=mutate):self.window.agent_local_import_button.click()
        self.assertIn('工单已变化',self.window.agent_local_summary.text());self.assertNotIn('local_evidence_reviews',InspectionTask.load(self.fixture.tp).to_report())
    def test_cancel_no_write(self):
        self.ready();before=self.fixture.tp.read_bytes()
        with patch.object(gui.QFileDialog,'getOpenFileName',return_value=('','')):self.window.agent_local_import_button.click()
        self.assertEqual(before,self.fixture.tp.read_bytes())
    def test_old_closed_state_disables_import(self):
        self.window._set_agent_state('completed_no_actionable_difference',self.fixture.tp)
        self.assertFalse(self.window.agent_local_import_button.isEnabled())
    def test_busy_worker_does_not_open_picker(self):
        self.ready()
        class Busy:
            def isRunning(self):return True
        self.window.port_worker=Busy()
        with patch.object(gui.QFileDialog,'getOpenFileName') as picker:self.window.agent_local_import_button.click();picker.assert_not_called()
        self.window.port_worker=None
        self.assertIn('分析尚在运行',self.window.agent_local_summary.text())
if __name__=='__main__':unittest.main()
