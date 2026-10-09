"""Headless actual Qt completion handler: no API, SAM or project writes."""
import copy,json,unittest
from pathlib import Path
import launch_llm_recheck_window_20261008 as window
from PyQt5.QtWidgets import QApplication

class QtPriorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def test_completion_preserves_local_results_and_records_priority(self):
        from probe_llm_recheck_planner_20261008 import CASES
        source=CASES['cabinet2']; report=json.loads(source.read_text(encoding='utf-8'))
        original=copy.deepcopy(report);fusion=report['sam3_fusion']
        ref=Path(fusion['reference_sam3']['output_dir'])/'mask_union.png'
        ins=Path(fusion['inspection_sam3']['output_dir'])/'mask_union.png'
        w=window.PlannedWindow()
        saved=[];w._write_report=lambda value:saved.append(copy.deepcopy(value))
        out=Path(__file__).resolve().parents[1]/'artifacts/priority_qt_test_output'
        w.current_output=out
        w._deepseek_pending={'output':out,'report':report,'reference_mask':ref,'inspection_mask':ins,'candidates':report['review_regions']}
        old=json.loads((Path(__file__).resolve().parents[1]/'artifacts/llm_recheck_planner_20261008/cabinet2.json').read_text(encoding='utf-8'))
        result=copy.deepcopy(old)
        for row in result['plan']['regions']:
            row.update(judgment='visible_change',suggested_priority='high',confidence='medium')
        result['answer_zh']=window.render_plan(result)
        w._finish_deepseek_mask_review(result)
        self.assertEqual(w.priority_table.rowCount(),2)
        self.assertEqual(w.priority_table.item(0,3).text(),'优先复核')
        self.assertEqual(saved[-1]['decision'],original['decision'])
        self.assertEqual(saved[-1]['review_regions'],original['review_regions'])
        self.assertIn('priority_adjustment',saved[-1]['external_mask_review'])
        stale=copy.deepcopy(result);stale['binding']={}
        w._finish_deepseek_mask_review(stale)
        self.assertEqual(w.priority_table.item(0,3).text(),'普通待复核')
        self.assertEqual(w.deepseek_answer_label.text(),'')
        self.assertEqual(saved[-1]['external_mask_review']['status'],'error')
        # Delayed results from different inputs must not alter the current view.
        w.current_output=out/'different'
        previous=len(saved);w._finish_deepseek_mask_review(result)
        self.assertEqual(len(saved),previous)
        w.close();w.deleteLater()

if __name__=='__main__':unittest.main()
