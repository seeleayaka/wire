import sys
from pathlib import Path
sys.path.insert(0,'E:/PythonProject10')
import unittest
import copy
from unittest.mock import patch
from independent_port_rescue import select_rescue,run_rescue
from inspection_agent.optional_port_crop_review import REFERENCE_SHA
from inspection_agent.port_tiling import tile_windows

ROI=dict(left=10,top=10,right=100,bottom=100)
def p(l=20,score=.8,coverage=1):return dict(left=l,top=20,right=l+10,bottom=30,confidence=score,class_id=0,valid_warp_fraction=coverage)

class RescueTests(unittest.TestCase):
    def test_one_hint_without_parent_link_or_verdict(self):
        hints,_=select_rescue(ROI,[],[p(),p(40,.9)],[])
        self.assertEqual(len(hints),1);self.assertEqual(hints[0]['box']['left'],40)
        self.assertIsNone(hints[0]['parent_index']);self.assertFalse(hints[0]['automatic_fault_verdict'])
    def test_static_reference_rejected(self):
        hints,audit=select_rescue(ROI,[],[p()],[p()])
        self.assertEqual(hints,[]);self.assertEqual(audit['static_reference_suppressed'],1)
    def test_roi_threshold_and_valid_warp(self):
        for item in (p(0),p(score=.25),p(coverage=.97)):
            self.assertEqual(select_rescue(ROI,[],[item],[])[0],[])
    def test_duplicate_does_not_add(self):
        self.assertEqual(select_rescue(ROI,[{'box':p()}],[p()],[])[0],[])
    def test_inputs_and_original_parents_unchanged_when_disabled(self):
        report={'review_regions':[{'bbox_xyxy':[1,2,3,4],'id':'original'}]};before=copy.deepcopy(report)
        result=run_rescue(report,project='E:/PythonProject10',reference_predictions={},roi=ROI)
        self.assertEqual(result['status'],'disabled');self.assertEqual(result['parents'],before['review_regions'])
        self.assertEqual(report,before)
    def test_bad_roi_fails_closed(self):
        result=run_rescue({},project='E:/PythonProject10',reference_predictions={},roi={**ROI,'left':True},enabled=True)
        self.assertEqual(result['status'],'fallback');self.assertEqual(result['rescue_hints'],[])
    def test_selection_does_not_mutate_predictions_or_existing(self):
        source=[p()];ref=[p(50)];old=[];before=copy.deepcopy((source,ref,old))
        select_rescue(ROI,old,source,ref);self.assertEqual((source,ref,old),before)
    def test_reference_identity_refuses_before_port_execution(self):
        with patch('independent_port_rescue.run_gui_port_review') as model:
            result=run_rescue({'reference':'unused'},project='unused',reference_predictions={},roi=ROI,enabled=True)
        self.assertEqual(result['fallback_reason'],'reference_prediction_identity_mismatch');model.assert_not_called()
    def test_wrap_preserves_real_sam_shape_and_bridge_failure(self):
        report={'reference':'unused','review_regions':[{'bbox_xyxy':[1,2,3,4],'tier':'dino_and_sam'}]}
        reference=dict(source_sha256=REFERENCE_SHA,source_shape=[2736,3648],
            windows=[list(w) for w in tile_windows(3648,2736)],edge_kept_predictions=[],merged_predictions=[])
        before=copy.deepcopy(report)
        with patch('independent_port_rescue.sha',return_value=REFERENCE_SHA):
            with patch('independent_port_rescue.run_gui_port_review',return_value=dict(status='fallback',fallback_reason='local_alignment_not_supported')):
                result=run_rescue(report,project='unused',reference_predictions=reference,roi=ROI,enabled=True)
        self.assertEqual(result['fallback_reason'],'local_alignment_not_supported')
        self.assertEqual(result['parents'],before['review_regions']);self.assertEqual(report,before)

if __name__=='__main__':unittest.main()
