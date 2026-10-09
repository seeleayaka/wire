"""No-region UI must start analysis; existing custom regions remain unchanged."""
import os,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(ROOT/'prototype'))
import assembly_auto_review_dino as dino
from PyQt5.QtWidgets import QApplication

class DefaultROIRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def check_start(self,rois,expected):
        window=dino.DINOReview()
        try:
            sample=ROOT/'examples/cabinets/2/2.png'
            window.reference.setText(str(sample));window.inspection.setText(str(sample))
            window.recipe={'check_rois':rois}
            with patch.object(dino,'InitialReviewWorker') as worker:
                window.run()
                self.assertEqual(worker.call_args.kwargs['check_rois'],expected)
                worker.return_value.start.assert_called_once()
                self.assertEqual(window.stage_label.text(),'正在读取图像')
            self.assertEqual(window.recipe['check_rois'],rois)
        finally:window.close()
    def test_empty_regions_default_to_whole_image(self):self.check_start([],[[0.0,0.0,1.0,1.0]])
    def test_custom_regions_remain_unchanged(self):self.check_start([[0.1,0.2,0.8,0.7]],[[0.1,0.2,0.8,0.7]])
    def test_missing_photographs_do_not_start_worker(self):
        window=dino.DINOReview()
        try:
            window.reference.setText('');window.inspection.setText('')
            with patch.object(dino,'InitialReviewWorker') as worker:
                window.run();worker.assert_not_called()
            self.assertEqual(window.stage_label.text(),'等待输入')
        finally:window.close()
if __name__=='__main__':unittest.main()
