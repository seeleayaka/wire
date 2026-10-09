import unittest,json
from unittest.mock import patch
from urllib.request import Request
import launch_llm_recheck_window_20261008 as ui
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage,QColor
from urllib.error import HTTPError

class CloudFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def test_connection_is_small_text_only_and_success_requires_reply(self):
        calls=[]
        def transport(req,timeout):calls.append(json.loads(req.data));return {'choices':[{'message':{'content':'OK'}}]}
        result=ui.probe_model_connection(ui.planner.backend.load_settings(),'test-local-token',transport)
        self.assertTrue(result['ok']);self.assertLessEqual(calls[0]['max_tokens'],16)
        self.assertNotIn('image',json.dumps(calls));self.assertEqual(calls[0]['thinking']['type'],'disabled')
        result=ui.probe_model_connection(ui.planner.backend.load_settings(),'test-local-token',lambda *a:{'choices':[]})
        self.assertFalse(result['ok'])
    def test_connection_errors_do_not_expose_credentials(self):
        for error,text in [(HTTPError('https://invalid',401,'secret',{},None),'密钥'),(HTTPError('https://invalid',429,'secret',{},None),'限流'),(TimeoutError('secret'),'超时')]:
            def transport(*args):raise error
            result=ui.probe_model_connection(ui.planner.backend.load_settings(),'secret',transport)
            self.assertFalse(result['ok']);self.assertIn(text,result['message']);self.assertNotIn('secret',result['message'])
    def test_missing_key_never_sends(self):
        with patch.object(ui.planner.backend,'_api_token',side_effect=RuntimeError('key configuration unavailable')),patch.object(ui.planner.backend,'_post_json') as send:
            result=ui.probe_model_connection(ui.planner.backend.load_settings())
            self.assertFalse(result['ok']);self.assertIn('密钥',result['message']);send.assert_not_called()
    def test_workbench_toggle_never_sends(self):
        w=ui.PlannedWindow()
        try:
            with patch.object(ui.planner.backend,'_post_json') as send:
                w.cloud_switch.setChecked(True);self.app.processEvents()
                self.assertTrue(w.cloud_dialog.isVisible());self.assertEqual(w.cloud_tabs.count(),2)
                w.cloud_dialog.close();w.show_cloud_workspace();self.assertTrue(w.cloud_dialog.isVisible())
                w.cloud_switch.setChecked(False);self.assertFalse(w.cloud_dialog.isVisible());send.assert_not_called()
        finally:w.close()
    def test_reading_maps_by_candidate_id_not_model_order(self):
        red=QImage(60,40,QImage.Format_RGB888);red.fill(QColor('red'))
        blue=QImage(60,40,QImage.Format_RGB888);blue.fill(QColor('blue'))
        result={'status':'ok','plan':{'summary_zh':'overview','review_order':['candidate_002','candidate_001'],
            'regions':[{'candidate_id':'candidate_001','observation_zh':'first','requested_checks':[],'question_zh':'q1'},
                       {'candidate_id':'candidate_002','observation_zh':'second','requested_checks':[],'question_zh':'q2'}]}}
        d=ui.ReviewReadingDialog(result,{'candidate_001':[red,red],'candidate_002':[blue,blue]})
        try:
            self.assertIn('second',d.opinion.toPlainText())
            self.assertEqual(d.picture_labels[0].pixmap().toImage().pixelColor(10,10),QColor('blue'))
            d.selector.setCurrentIndex(2);self.assertIn('first',d.opinion.toPlainText())
            self.assertEqual(d.picture_labels[0].pixmap().toImage().pixelColor(10,10),QColor('red'))
            d.font_size.setCurrentIndex(4);self.assertIn('26pt',d.opinion.styleSheet())
            self.assertEqual(d.opinion.document().defaultFont().pointSize(),26)
        finally:d.close()
    def test_reading_does_not_replace_missing_candidate_image(self):
        result={'status':'ok','plan':{'summary_zh':'overview','review_order':['candidate_001'],
            'regions':[{'candidate_id':'candidate_001','observation_zh':'first','requested_checks':[],'question_zh':'q1'}]}}
        d=ui.ReviewReadingDialog(result,{})
        try:self.assertIn('不可用',d.picture_labels[0].text())
        finally:d.close()
    def test_new_local_run_clears_previous_reading(self):
        w=ui.PlannedWindow()
        try:
            w._reading_result={'status':'ok'};w._reading_images={'candidate_001':[]};w.reading_button.setEnabled(True)
            with patch.object(ui.PlannedWindow.__bases__[0],'run') as run:
                w.run();run.assert_called_once()
            self.assertIsNone(w._reading_result);self.assertEqual(w._reading_images,{})
            self.assertFalse(w.reading_button.isEnabled())
        finally:w.close()
    def test_long_answer_can_scroll_to_last_line(self):
        w=ui.PlannedWindow()
        try:
            w.cloud_switch.setChecked(True)
            w.deepseek_answer_label.setText('\n'.join('复核建议 '+str(i) for i in range(180))+'\n最后一条建议')
            w.cloud_tabs.setCurrentIndex(1)
            w.show();self.app.processEvents()
            scroll=w.deepseek_answer_scroll
            self.assertLessEqual(scroll.height(),260)
            self.assertGreater(scroll.verticalScrollBar().maximum(),0)
            scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
            self.app.processEvents()
            self.assertIn('最后一条建议',w.deepseek_answer_label.text())
            self.assertEqual(scroll.verticalScrollBar().value(),scroll.verticalScrollBar().maximum())
        finally:w.close()
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
