import json,unittest
from unittest.mock import patch
import launch_llm_recheck_window_20261008 as ui
import deepseek_thinking_options as options
from PyQt5.QtWidgets import QApplication
from urllib.request import Request

class ThinkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def test_modes_are_transmitted_and_budget_bounded(self):
        base=ui.planner.backend.load_settings()
        for mode in options.MODES:
            cfg=options.configured_settings(base,mode);self.assertLessEqual(cfg.max_tokens,8192);self.assertLessEqual(cfg.timeout_seconds,120)
            calls=[]
            def transport(req,timeout):calls.append(json.loads(req.data));return {'choices':[{'message':{'content':'{}','reasoning_content':'private'},'finish_reason':'stop'}]}
            diagnostics={};sender=options.thinking_sender(mode,transport,diagnostics=diagnostics)
            sender(Request('https://example.invalid',data=b'{"messages":[]}'),cfg.timeout_seconds)
            self.assertEqual(calls[0]['thinking']['type'],'disabled' if mode=='disabled' else 'enabled')
            self.assertEqual(calls[0].get('reasoning_effort'),None if mode=='disabled' else mode)
            self.assertTrue(diagnostics['reasoning_present']);self.assertNotIn('private',json.dumps(diagnostics))
    def test_invalid_modes_fail_closed(self):
        with self.assertRaises(ValueError):options.configured_settings(ui.planner.backend.load_settings(),'unbounded')
    def test_settings_only_appear_when_enabled_and_no_auto_send(self):
        w=ui.PlannedWindow()
        try:
            self.assertFalse(w.cloud_switch.isChecked());self.assertTrue(w.cloud_body.isHidden())
            with patch.object(ui.priority,'run') as network:
                w.cloud_switch.setChecked(True);self.assertFalse(w.cloud_body.isHidden())
                w.cloud_switch.setChecked(False);w.start_deepseek_mask_review();network.assert_not_called()
            self.assertEqual(w.thinking_mode.count(),4)
            self.assertEqual(w.deepseek_api_input.echoMode(),w.deepseek_api_input.Password)
        finally:w.close()
    def test_key_save_uses_existing_local_store_and_clears_input(self):
        w=ui.PlannedWindow()
        try:
            w.deepseek_api_input.setText('local-test-not-real-credential')
            with patch.object(ui.planner.backend,'save_local_api_key') as save:w._save_cloud_key();save.assert_called_once()
            self.assertEqual(w.deepseek_api_input.text(),'');self.assertTrue(w.deepseek_saved_key_available)
        finally:w.close()
    def test_worker_snapshots_selected_strength(self):
        w=ui.PlannedWindow()
        try:
            w.thinking_mode.setCurrentIndex(w.thinking_mode.findData('low'))
            worker=ui.PlannedWorker(reference_mask_path=None,inspection_mask_path=None,candidates=[],question_zh='review',parent=w)
            w.thinking_mode.setCurrentIndex(w.thinking_mode.findData('high'))
            self.assertEqual(worker.thinking_mode,'low');worker.deleteLater()
        finally:
            w.thinking_mode.setCurrentIndex(w.thinking_mode.findData('low'));w.close()
if __name__=='__main__':unittest.main()
