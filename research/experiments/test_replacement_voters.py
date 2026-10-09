from copy import deepcopy
import unittest
from replacement_voters import validate_views, check_prefix


class ReplacementTests(unittest.TestCase):
    def setUp(self):
        self.roles = dict(teacher='a'*64, student='b'*64, feature='c'*64)
        self.views = [dict(weight_sha256=d, source_sha256='d'*64,
                          predictions=dict(source_shape=[100,200], merged_predictions=[
                              dict(class_id=0, confidence=.8, box_xyxy=[20,20,40,40])]))
                      for d in self.roles.values()]

    def test_three_roles_and_multiple_views(self):
        views = self.views + [deepcopy(self.views[1])]
        self.assertEqual(validate_views(views,self.roles,'d'*64,[100,200]),views)

    def test_replaced_student_rejected(self):
        views = self.views + [dict(self.views[1],weight_sha256='e'*64)]
        with self.assertRaises(ValueError): validate_views(views,self.roles,'d'*64,[100,200])

    def test_duplicate_roles_rejected(self):
        with self.assertRaises(ValueError): validate_views(self.views,dict(self.roles,student='a'*64),'d'*64,[100,200])

    def test_missing_role_rejected(self):
        with self.assertRaises(ValueError): validate_views(self.views[:2],self.roles,'d'*64,[100,200])

    def test_source_and_shape_fail_closed(self):
        for change in [dict(source_sha256='e'*64),dict(predictions=dict(source_shape=[200,100],merged_predictions=[]))]:
            views = deepcopy(self.views); views[0].update(change)
            with self.assertRaises(ValueError): validate_views(views,self.roles,'d'*64,[100,200])

    def test_invalid_rows_rejected(self):
        for change in [dict(confidence=float('nan')),dict(class_id=2),dict(box_xyxy=[0,0,float('inf'),30])]:
            views = deepcopy(self.views);views[0]['predictions']['merged_predictions'][0].update(change)
            with self.assertRaises(ValueError): validate_views(views,self.roles,'d'*64,[100,200])

    def test_inputs_preserved(self):
        result=validate_views(self.views,self.roles,'d'*64,[100,200]);result[0]['predictions']['source_shape'][0]=1
        self.assertEqual(self.views[0]['predictions']['source_shape'],[100,200])

    def test_exact_double_prefix(self):
        old=dict(primary=['p'],all_predictions=['p','old'])
        research=dict(primary=['p'],all_predictions=['p','old','r'])
        trial=dict(primary=['p'],all_predictions=['p','old','r','new'])
        self.assertTrue(check_prefix(old,research,trial))
        for bad in [dict(trial,all_predictions=['p','r','old','new']),dict(trial,primary=[]),dict(trial,all_predictions=['p','old','r',1,2,3,4])]:
            with self.assertRaises(ValueError):check_prefix(old,research,bad)


if __name__=='__main__': unittest.main()
