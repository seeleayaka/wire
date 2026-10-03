import copy
import unittest
from unittest.mock import patch
from inspection_agent.independent_port_rescue import review_rois, select_rescue, run_independent_port_rescue

ROI = dict(left=10, top=10, right=100, bottom=100)
def p(left=20, confidence=.8, coverage=1, cls=0):
    return dict(left=left,top=20,right=left+10,bottom=30,confidence=confidence,
                class_id=cls,valid_warp_fraction=coverage)

class RescueTests(unittest.TestCase):
    def test_disabled_no_model(self):
        report={'review_regions':[{'bbox_xyxy':[1,2,3,4]}]};before=copy.deepcopy(report)
        with patch('inspection_agent.independent_port_rescue.run_gui_port_review') as model:
            result=run_independent_port_rescue(report,project='unused')
        model.assert_not_called();self.assertEqual(result['status'],'disabled');self.assertEqual(before,report)
    def test_legacy_report_missing_roi_rejected(self):
        with patch('inspection_agent.independent_port_rescue.run_gui_port_review') as model:
            result=run_independent_port_rescue({},project='unused',enabled=True)
        model.assert_not_called();self.assertEqual(result['fallback_reason'],'analysis_roi_provenance_missing')
    def test_bad_rois(self):
        for roi in ([True,.1,.9,.9],[0,0,2,1],[0,0,0,1],[0,0,float('nan'),1],['a',0,1,1]):
            with self.assertRaises(ValueError):review_rois({'analysis_check_rois':[roi]})
    def test_multiple_rois_one_global_hint(self):
        hints,_=select_rescue([ROI,dict(left=100,top=10,right=200,bottom=100)],[],[p(),p(120,.9)],[])
        self.assertEqual(len(hints),1);self.assertEqual(hints[0]['box']['left'],120)
        self.assertIsNone(hints[0]['parent_index']);self.assertFalse(hints[0]['automatic_fault_verdict'])
    def test_static_reference_removed(self):
        hints,audit=select_rescue([ROI],[],[p()],[p()])
        self.assertEqual(hints,[]);self.assertEqual(audit['static_reference_suppressed'],1)
    def test_different_class_not_static(self):
        self.assertEqual(len(select_rescue([ROI],[],[p()],[p(cls=1)])[0]),1)
    def test_threshold_roi_validity(self):
        for prediction in (p(left=0),p(confidence=.25),p(coverage=.97)):
            self.assertEqual(select_rescue([ROI],[],[prediction],[])[0],[])
    def test_existing_hint_deduplicated(self):
        self.assertEqual(select_rescue([ROI],[{'box':p()}],[p()],[])[0],[])
    def test_selection_immutable(self):
        args=([ROI],[],[p()],[p(50)]);before=copy.deepcopy(args)
        select_rescue(*args);self.assertEqual(args,before)
    def test_bridge_rejection_preserves_sam_candidates(self):
        report={'analysis_check_rois':[[0,0,1,1]],'review_regions':[{'bbox_xyxy':[1,2,3,4]}]}
        with patch('inspection_agent.independent_port_rescue.run_gui_port_review',return_value={
                'status':'fallback','fallback_reason':'local_alignment_not_supported'}):
            result=run_independent_port_rescue(report,project='unused',enabled=True)
        self.assertEqual(result['parents'],report['review_regions']);self.assertEqual(result['rescue_hints'],[])
    def test_fresh_reference_uses_same_bridge_and_identity(self):
        report=dict(analysis_check_rois=[[0,0,1,1]],review_regions=[{'bbox_xyxy':[1,2,3,4]}],
                    inspection='source',reference='reference',alignment={},image_fingerprints={})
        before=copy.deepcopy(report)
        source=dict(status='applied',source_sha256='source_sha',reference_sha256='ref_sha',aligned_predictions=[p()])
        ref=dict(status='applied',aligned_predictions=[])
        with patch('inspection_agent.independent_port_rescue.run_gui_port_review',side_effect=[source,ref]) as bridge:
            with patch('inspection_agent.optional_port_crop_review.sha',return_value='source_sha'):
                result=run_independent_port_rescue(report,project='unused',enabled=True)
        reference_call=bridge.call_args_list[1].args[0]
        self.assertEqual(reference_call['inspection'],'reference')
        self.assertEqual(reference_call['alignment']['source_to_reference_homography'],[[1,0,0],[0,1,0],[0,0,1]])
        self.assertEqual(result['status'],'applied');self.assertEqual(result['parents'],before['review_regions'])
        self.assertEqual(report,before)
    def test_source_changed_while_reference_running(self):
        report=dict(analysis_check_rois=[[0,0,1,1]],inspection='source',reference='reference',alignment={},image_fingerprints={})
        source=dict(status='applied',source_sha256='old',reference_sha256='ref',aligned_predictions=[p()])
        with patch('inspection_agent.independent_port_rescue.run_gui_port_review',side_effect=[source,dict(status='applied',aligned_predictions=[])]):
            with patch('inspection_agent.optional_port_crop_review.sha',return_value='new'):
                result=run_independent_port_rescue(report,project='unused',enabled=True)
        self.assertEqual(result['rescue_hints'],[]);self.assertEqual(result['status'],'fallback')

if __name__=='__main__':unittest.main()
