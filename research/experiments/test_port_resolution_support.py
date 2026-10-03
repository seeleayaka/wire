import copy,unittest
from test_teacher_student_port_policy import case,row,fixtures
from teacher_student_port_policy import merge
from port_resolution_support import append_resolution,ResolutionModel

class ResolutionTests(unittest.TestCase):
    def test_preserves_whole_accepted_selection(self):
        teacher,alternative=fixtures();current=merge(teacher,case([]));before=copy.deepcopy(current)
        output=append_resolution(teacher,current,alternative)
        self.assertEqual(output['all_predictions'][:len(current['all_predictions'])],current['all_predictions'])
        self.assertEqual(len(output['resolution_additions']),1);self.assertEqual(current,before)
    def test_full_budget_no_new_cues(self):
        teacher,alternative=fixtures(True);current=merge(teacher,case([]))
        self.assertEqual(append_resolution(teacher,current,alternative)['resolution_additions'],[])
    def test_dedup_against_accepted_student_and_feature(self):
        teacher,alternative=fixtures();current=merge(teacher,alternative)
        self.assertEqual(append_resolution(teacher,current,alternative)['resolution_additions'],[])
    def test_identity_and_geometry_fail_safe(self):
        for change in ('source','geometry'):
            teacher,alternative=fixtures();current=merge(teacher,case([]))
            if change=='source':alternative['source_sha256']='changed'
            else:alternative['predictions']['source_shape']=[1,1]
            output=append_resolution(teacher,current,alternative)
            self.assertEqual(output['all_predictions'],current['all_predictions']);self.assertIsNotNone(output['resolution_fallback_reason'])
    def test_boundaries_unchanged(self):
        teacher,alternative=fixtures();current=merge(teacher,case([]));alternative['predictions']['merged_predictions'][0]['confidence']=.75
        self.assertEqual(append_resolution(teacher,current,alternative)['resolution_additions'],[])
    def test_two_views_still_required(self):
        teacher,alternative=fixtures();current=merge(teacher,case([]));alternative['predictions']['edge_kept_predictions'][1]['source_tile']=0
        self.assertEqual(append_resolution(teacher,current,alternative)['resolution_additions'],[])
    def test_wrapper_changes_only_input_resolution(self):
        calls=[]
        class Model:
            def predict(self,*args,**kwargs):calls.append((args,kwargs));return ['result']
        wrapped=ResolutionModel(Model(),1280,lambda:None)
        self.assertEqual(wrapped.predict('image',imgsz=960,conf=.001,iou=.7,max_det=300),['result'])
        self.assertEqual(calls,[(('image',),dict(imgsz=1280,conf=.001,iou=.7,max_det=300))])
    def test_reject_unsupported_resolution(self):
        with self.assertRaises(ValueError):ResolutionModel(None,961,lambda:None)

if __name__=='__main__':unittest.main()
