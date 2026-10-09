import unittest,json
from unittest.mock import patch
from urllib.request import Request
import launch_llm_recheck_window_20261008 as ui
from PyQt5.QtWidgets import QApplication

class CloudFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def test_busy_and_completion_feedback(self):
        w=ui.PlannedWindow()
        try:
            w._on_cloud_progress('请求正在发送，等待大模型判断')
            self.assertTrue(w.cloud_spinner._timer.isActive());self.assertTrue(w.cloud_elapsed.isActive())
            self.assertNotIn('发送成功',w.cloud_status.text())
            w._finish_cloud_feedback(True)
            self.assertFalse(w.cloud_spinner._timer.isActive());self.assertFalse(w.cloud_elapsed.isActive())
            self.assertIn('发送成功',w.cloud_status.text())
            w._finish_cloud_feedback(False);self.assertNotIn('发送成功',w.cloud_status.text())
        finally:w.close()
    def test_transport_stage_follows_actual_response(self):
        worker=ui.PlannedWorker(reference_mask_path=None,inspection_mask_path=None,candidates=[],question_zh='test')
        phases=[];worker.progress.connect(phases.append)
        def run(*args,**kwargs):
            kwargs['sender'](Request('https://example.invalid',data=json.dumps({'messages':[]}).encode()),45)
            return {'status':'ok','plan':{}}
        with patch.object(ui.priority,'run',side_effect=run),patch.object(ui,'render_plan',return_value='ok'),patch.object(ui.planner.backend,'_post_json',return_value={'choices':[{'message':{'content':'{}'}}]}):worker.run()
        self.assertIn('请求正在发送',phases[1]);self.assertIn('服务已响应',phases[2]);worker.deleteLater()
    def test_transport_failure_never_claims_sent_success(self):
        worker=ui.PlannedWorker(reference_mask_path=None,inspection_mask_path=None,candidates=[],question_zh='test')
        phases=[];worker.progress.connect(phases.append)
        def run(*args,**kwargs):kwargs['sender'](Request('https://example.invalid',data=json.dumps({'messages':[]}).encode()),45)
        with patch.object(ui.priority,'run',side_effect=run),patch.object(ui.planner.backend,'_post_json',side_effect=OSError('offline')):worker.run()
        self.assertFalse(any('请求成功' in phase for phase in phases));worker.deleteLater()
    def test_optional_tabs_keep_auto_navigation(self):
        w=ui.PlannedWindow()
        try:
            for view,title in [(w.port_hint_view,'端口局部提示'),(w.port_comparison_panel,'端口局部对比'),(w.rescue_view,'独立端口补漏')]:
                w.tabs.setCurrentWidget(view)
                self.assertEqual(w.tabs.tabText(w.tabs.currentIndex()),title)
                self.assertTrue(any('可选工具' in label.text() or '先生成端口局部提示' in label.text() for label in w.tabs.currentWidget().findChildren(ui.QLabel)))
        finally:w.close()
if __name__=='__main__':unittest.main()
