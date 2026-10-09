import unittest
from copy import deepcopy
import json
import tempfile
from pathlib import Path
from audit_allport480_source import (iou,score,audit_views,audit_probabilities,tile_windows,
    audit_semantic_reuse,sha,DATA,HEAD_SHA,REFERENCE_SHA)
from exact_paired_score_cache import binding


class SourceScoreTests(unittest.TestCase):
    def test_class_aware_and_duplicate_burden(self):
        targets=[dict(class_id=0,box=[10,10,20,20]),dict(class_id=1,box=[30,30,40,40])]
        rows=[dict(class_id=0,box_xyxy=[10,10,20,20]),dict(class_id=0,box_xyxy=[10,10,20,20]),dict(class_id=0,box_xyxy=[30,30,40,40])]
        metric,hits=score(rows,targets)
        self.assertEqual(metric,dict(tp=1,unmatched=2,fn=1,predictions=3,targets=2));self.assertEqual(hits,{0})

    def test_normal_empty_abstention_not_recognition_metric(self):
        self.assertEqual(score([],[])[0],dict(tp=0,unmatched=0,fn=0,predictions=0,targets=0))

    def test_exact_iou_boundary(self):
        self.assertEqual(iou([0,0,20,10],[0,0,10,10]),.5)
        self.assertEqual(score([dict(class_id=0,box_xyxy=[0,0,20,10])],[dict(class_id=0,box=[0,0,10,10])])[1],{0})


class ActualEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.roles=dict(teacher='a'*64,student='b'*64,feature='c'*64)
        self.source='d'*64;self.shape=[2736,3648]
        grids=[[[0,0,3648,2736]]]+[[list(w) for w in tile_windows(3648,2736,t,s)] for t,s in [(1280,960),(960,720)]]
        self.views=[dict(weight_sha256=d,source_sha256=self.source,predictions=dict(source_shape=self.shape,
            windows=w,edge_kept_predictions=[],merged_predictions=[])) for d in self.roles.values() for w in grids]

    def check(self,views=None):
        return audit_views(self.views if views is None else views,self.roles,self.source,self.shape)

    def test_all_empty_actual_recipes_are_valid_negatives(self):
        before=deepcopy(self.views)
        self.assertEqual(self.check(),9);self.assertEqual(before,self.views)

    def test_placeholder_empty_is_not_negative_inference(self):
        rows=deepcopy(self.views);del rows[1]['predictions']['windows']
        with self.assertRaises(ValueError):self.check(rows)

    def test_four_views_of_one_weight_cannot_replace_another_role(self):
        rows=deepcopy(self.views);rows[3]['weight_sha256']=rows[0]['weight_sha256']
        with self.assertRaises(ValueError):self.check(rows)

    def test_missing_or_duplicate_recipe_rejected(self):
        for rows in [self.views[:-1],self.views+[deepcopy(self.views[0])]]:
            with self.subTest(count=len(rows)),self.assertRaises(ValueError):self.check(rows)

    def test_wrong_source_or_frame_rejected(self):
        for key,value in [('source_sha256','e'*64),('weight_sha256','f'*64)]:
            rows=deepcopy(self.views);rows[1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.check(rows)
        rows=deepcopy(self.views);rows[1]['predictions']['source_shape']=[3648,2736]
        with self.assertRaises(ValueError):self.check(rows)

    def test_native_merge_tampering_rejected(self):
        rows=deepcopy(self.views);rows[1]['predictions']['merged_predictions']=[dict(box_xyxy=[10,10,20,20],class_id=0,confidence=.9)]
        with self.assertRaises(ValueError):self.check(rows)

    def test_finite_ordered_bounded_actual_boxes(self):
        for box in [[20,10,10,20],[10,10,10,20],[-1,10,20,20],[10,10,4000,20],[True,10,20,20],[10,float('nan'),20,20]]:
            rows=deepcopy(self.views);rows[0]['predictions']['merged_predictions']=[dict(box_xyxy=box,class_id=0,confidence=.9)]
            with self.subTest(box=box),self.assertRaises(ValueError):self.check(rows)

    def test_invalid_confidence_and_class(self):
        for cls,confidence in [(True,.9),(.0,.9),(2,.9),(0,float('inf')),(0,True),(0,-.1)]:
            rows=deepcopy(self.views);rows[0]['predictions']['merged_predictions']=[dict(box_xyxy=[10,10,20,20],class_id=cls,confidence=confidence)]
            with self.subTest(cls=cls,confidence=confidence),self.assertRaises(ValueError):self.check(rows)

    def test_probability_vectors_and_counts(self):
        audit_probabilities([{}],[[.1,.8,.1]])
        for rows in [[],[[.1,.8]],[[.1,.8,.2]],[[True,0,0]],[[float('nan'),0,1]]]:
            with self.subTest(rows=rows),self.assertRaises(ValueError):audit_probabilities([{}],rows)


class SemanticCacheAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        folder=Path(self.temp.name);(folder/'train').mkdir()
        self.path=folder/'train/fixture_predictions.json'
        self.runtime=dict(encoder='e'*64,head=HEAD_SHA)
        self.alignment=dict(alignment_quality={'reliable':True},source_to_reference_homography=[[1,0,0],[0,1,0],[0,0,1]])
        self.row=dict(class_id=0,box_xyxy=[10,10,20,20]);self.prob=[.01,.985,.005];self.source='a'*64
        self.parent=dict(head_sha256=HEAD_SHA,image='fixture.JPG',source_sha256=self.source,alignment=self.alignment,
            proposals=[self.row],probabilities=[self.prob])
        self.path.write_text(json.dumps(self.parent),encoding='utf-8')
        reportpath=folder/'report.json'
        reportpath.write_text(json.dumps(dict(runtime=self.runtime,pins={str(DATA/'images/train01/normal_073.JPG'):REFERENCE_SHA})),encoding='utf-8')
        self.report=dict(runtime=self.runtime,pins={str(self.path):sha(self.path),str(reportpath):sha(reportpath)})
        self.digest=binding(self.source,REFERENCE_SHA,self.runtime,self.alignment,[2736,3648])
        self.case=dict(image='fixture.JPG',source_sha256=self.source,alignment=self.alignment,proposals=[self.row],probabilities=[self.prob],
            semantic_score_reuse=dict(source_case_path=str(self.path),source_case_sha256=sha(self.path),old_input_binding=self.digest,
                new_input_binding=self.digest,reused_count=1,fresh_indices=[],same_classifier_not_an_extra_vote=True))

    def test_exact_parent_replays_one_cached_probability(self):
        self.assertEqual(audit_semantic_reuse(self.case,self.report,True),(1,0))

    def test_missing_required_provenance_rejected(self):
        self.case['semantic_score_reuse']=None
        with self.assertRaises(ValueError):audit_semantic_reuse(self.case,self.report,True)

    def test_changed_parent_output_rejected(self):
        self.path.write_text('{}',encoding='utf-8')
        with self.assertRaises(ValueError):audit_semantic_reuse(self.case,self.report,True)

    def test_changed_cached_probability_rejected(self):
        self.case['probabilities']=[[.02,.975,.005]]
        with self.assertRaises(ValueError):audit_semantic_reuse(self.case,self.report,True)

    def test_forged_cache_count_binding_or_vote_rejected(self):
        for key,value in [('reused_count',2),('fresh_indices',[0]),('new_input_binding','f'*64),('same_classifier_not_an_extra_vote',False)]:
            case=deepcopy(self.case);case['semantic_score_reuse'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):audit_semantic_reuse(case,self.report,True)


if __name__=='__main__':unittest.main()
