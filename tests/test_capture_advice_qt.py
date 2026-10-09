import copy,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'models/dinov2'),str(ROOT/'prototype')]
import assembly_auto_review_dino as gui
from PyQt5.QtWidgets import QApplication

class CaptureAdviceQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def test_local_guidance_visible_and_persisted_without_llm(self):
        window=gui.DINOReview()
        try:
            with tempfile.TemporaryDirectory() as directory:
                window.current_output=Path(directory)
                report={'decision':'possible_difference_manual_review','review_regions':[{'tier':'dino_and_sam'}],
                        'alignment':{'alignment_quality':{'reliable':True}}}
                before=copy.deepcopy(report)
                with patch.object(gui.deepseek_mask_review,'_post_json',side_effect=AssertionError('Unexpected network')):
                    window._write_report(report)
                self.assertIsNone(window._deepseek_pending)
                self.assertIn('无需大模型',window.capture_advice_label.text())
                self.assertIn('左侧或右侧',window.capture_advice_label.text())
                saved=json.loads((Path(directory)/'report.json').read_text(encoding='utf-8'))
                self.assertFalse(saved['capture_advice']['requires_llm'])
                self.assertEqual(saved['review_regions'],before['review_regions'])
                self.assertEqual(report,before)
        finally:window.close()
    def test_alignment_uncertain_shows_same_angle_without_sam_or_llm(self):
        window=gui.DINOReview()
        try:
            with tempfile.TemporaryDirectory() as directory:
                report={'decision':'alignment_uncertain_manual_review',
                        'alignment':{'alignment_quality':{'reliable':False}}}
                with (patch.object(window,'_create_agent_task'),
                      patch.object(gui.Sam3FusionWorker,'start',side_effect=AssertionError('Unexpected SAM')),
                      patch.object(gui.deepseek_mask_review,'_post_json',side_effect=AssertionError('Unexpected network'))):
                    window._finish_initial_review({'status':'alignment_uncertain','output':directory,'report':report})
                self.assertIn('定位不可靠',window.capture_advice_label.text())
        finally:window.close()

if __name__=='__main__':unittest.main()
