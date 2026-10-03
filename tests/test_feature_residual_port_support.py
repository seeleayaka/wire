import copy
import unittest
from unittest.mock import patch
import numpy as np
from inspection_agent.feature_residual_port_support import (
    residual_candidates,append_feature,run_feature_residual_review,WEIGHT_SHA)


def row(x,score=.9,tile=0):return dict(box_xyxy=[x,300,x+40,340],confidence=score,class_id=0,source_tile=tile)
def case(rows):return dict(image='test.JPG',source_sha256='same',weight_sha256='test',zoom_evidence=[],
    predictions=dict(source_shape=[2736,3648],merged_predictions=copy.deepcopy(rows),edge_kept_predictions=copy.deepcopy(rows)))
def cases():
    teacher=case([row(100+i*60,.99-i*.005) for i in range(5)]+[row(1200,.3),row(1500,.3)])
    old=case([row(1200)]);old['predictions']['edge_kept_predictions'].append(row(1200,.85,1))
    new=case([row(1500)]);new['predictions']['edge_kept_predictions'].append(row(1500,.85,1))
    return teacher,old,new
def base():return dict(status='applied',rescue_hints=[],supplementary_hints=[],existing_hints=[],
    source_evidence=dict(predictions=dict(source_shape=[2736,3648])),
    reference_evidence=dict(predictions=dict(source_shape=[2736,3648]),aligned_predictions=[]),
    analysis_rois=[dict(left=0,top=0,right=3648,bottom=2736)])


class FeatureResidualTests(unittest.TestCase):
    def test_old_prefix_and_inputs_unchanged(self):
        from inspection_agent.teacher_student_port_support import complementary_candidates
        teacher,old,new=cases();before=copy.deepcopy((teacher,old,new));accepted=complementary_candidates(teacher,old)
        result=residual_candidates(teacher,old,new)
        self.assertEqual(result['all_predictions'][:len(accepted['all_predictions'])],accepted['all_predictions'])
        self.assertEqual((teacher,old,new),before);self.assertEqual(len(result['feature_additions']),1)
    def test_identity_and_geometry_fail_safe(self):
        for change in ('identity','geometry'):
            teacher,old,new=cases()
            if change=='identity':new['source_sha256']='other'
            else:new['predictions']['source_shape']=[100,100]
            result=residual_candidates(teacher,old,new)
            self.assertEqual(result['feature_additions'],[]);self.assertEqual(len(result['all_predictions']),6)
    def test_deduplication(self):
        teacher,old,new=cases();self.assertEqual(residual_candidates(teacher,old,old)['feature_additions'],[])
    def test_original_extra_slots_shared(self):
        teacher,old,new=cases();rows=[row(100+i*60,.99-i*.005) for i in range(10)]
        teacher['predictions']['merged_predictions']=rows+[row(1500,.3)]
        teacher['predictions']['edge_kept_predictions']=rows+[dict(r,source_tile=1) for r in rows]
        output=residual_candidates(teacher,old,new)
        self.assertEqual(len(output['all_predictions']),10);self.assertEqual(output['feature_additions'],[])
    def test_reference_veto_and_metadata(self):
        cue=row(100);ref=dict(left=100,top=300,right=140,bottom=340,class_id=0,confidence=.9,valid_warp_fraction=1)
        self.assertEqual(append_feature(base(),[cue],[ref],np.eye(3))[1],[])
        added=append_feature(base(),[cue],[],np.eye(3))[1]
        self.assertEqual(added[0]['student_weight_sha256'],WEIGHT_SHA)
        self.assertEqual(added[0]['evidence_tier'],'feature_residual_manual_review')
    def test_roi_and_invalid_warp(self):
        old=base();old['analysis_rois']=[dict(left=500,top=500,right=900,bottom=900)]
        self.assertEqual(append_feature(old,[row(100)],[],np.eye(3))[1],[])
        matrix=np.eye(3);matrix[0,2]=4000
        self.assertEqual(append_feature(base(),[row(100)],[],matrix)[1],[])
    def test_full_supplement_budget(self):
        old=base();old['supplementary_hints']=[dict(box=dict(left=700+i*80,top=900,right=750+i*80,bottom=950)) for i in range(5)]
        output,added=append_feature(old,[row(100)],[],np.eye(3))
        self.assertEqual(added,[]);self.assertEqual(output['supplementary_hints'],old['supplementary_hints'])
    def test_off_never_reads_feature_files(self):
        with patch('inspection_agent.feature_residual_port_support.run_teacher_student_review',return_value=base()),patch(
            'inspection_agent.feature_residual_port_support.residual_runtime_fingerprint',side_effect=RuntimeError('no new reads')):
            output=run_feature_residual_review({},project='unused')
            self.assertFalse(output['feature_residual_policy']['enabled'])
    def test_base_failure_does_not_invoke_feature(self):
        old=base();old['status']='fallback'
        with patch('inspection_agent.feature_residual_port_support.run_teacher_student_review',return_value=old),patch(
            'inspection_agent.feature_residual_port_support.residual_runtime_fingerprint',side_effect=RuntimeError('no reads')):
            self.assertEqual(run_feature_residual_review({},project='unused',feature_enabled=True)['status'],'fallback')
    def test_unavailable_accepted_student_returns_safely(self):
        with patch('inspection_agent.feature_residual_port_support.run_teacher_student_review',return_value=base()):
            output=run_feature_residual_review({},project='unused',feature_enabled=True)
            self.assertEqual(output['feature_residual_policy']['fallback_reason'],'accepted_pair_not_available')


if __name__=='__main__':unittest.main()
