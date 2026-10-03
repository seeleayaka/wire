import copy
import unittest
from unittest.mock import patch
import numpy as np

from port_residual_feature_backend import append_feature, run_residual_review, WEIGHT_SHA


def base():
    return dict(status='applied',rescue_hints=[],supplementary_hints=[],existing_hints=[],
        source_evidence=dict(predictions=dict(source_shape=[2736,3648])),
        reference_evidence=dict(predictions=dict(source_shape=[2736,3648]),aligned_predictions=[]),
        analysis_rois=[dict(left=0,top=0,right=3648,bottom=2736)])


def cue(x=100):
    return dict(box_xyxy=[x,100,x+40,140],class_id=0,confidence=.9)


class BackendTests(unittest.TestCase):
    def test_provenance_and_preservation(self):
        old=base(); before=copy.deepcopy(old)
        output,added=append_feature(old,[cue()],[],np.eye(3))
        self.assertEqual(old,before); self.assertEqual(len(added),1)
        self.assertEqual(added[0]['student_weight_sha256'],WEIGHT_SHA)
        self.assertEqual(added[0]['evidence_tier'],'feature_residual_manual_review')
        self.assertEqual(output['rescue_hints'],old['rescue_hints'])

    def test_reference_static_veto(self):
        ref=dict(left=100,top=100,right=140,bottom=140,class_id=0,confidence=.9,valid_warp_fraction=1)
        self.assertEqual(append_feature(base(),[cue()],[ref],np.eye(3))[1],[])

    def test_roi_and_warp(self):
        old=base(); old['analysis_rois']=[dict(left=500,top=500,right=900,bottom=900)]
        self.assertEqual(append_feature(old,[cue()],[],np.eye(3))[1],[])
        matrix=np.eye(3);matrix[0,2]=4000
        self.assertEqual(append_feature(base(),[cue()],[],matrix)[1],[])

    def test_full_budget_preserved(self):
        old=base();old['supplementary_hints']=[dict(box=dict(left=700+i*80,top=900,right=750+i*80,bottom=950)) for i in range(5)]
        output,added=append_feature(old,[cue()],[],np.eye(3))
        self.assertEqual(added,[]);self.assertEqual(output['supplementary_hints'],old['supplementary_hints'])

    def test_disabled_does_not_read_new_weight(self):
        with patch('port_residual_feature_backend.run_teacher_student_review',return_value=base()),patch('port_residual_feature_backend.sha',side_effect=RuntimeError('no reads')):
            self.assertFalse(run_residual_review({})['feature_residual_policy']['enabled'])

    def test_unavailable_accepted_branch_returns_safely(self):
        old=base();old['teacher_student_policy']=dict(fallback_reason='bad_manifest')
        with patch('port_residual_feature_backend.run_teacher_student_review',return_value=old),patch('port_residual_feature_backend.sha',side_effect=RuntimeError('no reads')):
            result=run_residual_review({},feature_enabled=True)
            self.assertEqual(result['feature_residual_policy']['fallback_reason'],'accepted_pair_not_available')


if __name__=='__main__':unittest.main()
