import unittest
from audit_allport_missing_class_coverage import classify_coverage


class MissingClassTests(unittest.TestCase):
    def check(self,entries):
        target=dict(class_id=0,box=[0,0,10,10]);roles=dict(teacher='a',student='b',feature='c')
        views=[dict(weight_sha256=d,predictions=dict(merged_predictions=rows)) for d,rows in entries]
        return classify_coverage(target,views,roles)

    def row(self,cls=0,conf=.99,box=None):return dict(class_id=cls,confidence=conf,box_xyxy=box or [0,0,10,10])

    def test_same_weight_many_views_only_one_support(self):
        r=self.check([('a',[self.row()])]*4)
        self.assertEqual(r['distinct_roles_with_same_IoU50'],1)

    def test_correct_and_wrong_class_separate(self):
        r=self.check([('a',[self.row(1)]),('b',[self.row()])])
        self.assertEqual(r['distinct_roles_with_same_IoU50'],1);self.assertEqual(r['distinct_roles_with_other_IoU50'],1)
        self.assertEqual(r['reason'],'some_correct_class_geometry_but_missing_other_voters')

    def test_wrong_class_is_not_missing_object(self):
        self.assertEqual(self.check([('a',[self.row(1)])])['reason'],'localized_port_only_wrong_detector_class')

    def test_floor_and_absent_geometry(self):
        self.assertEqual(self.check([('a',[self.row(conf=.05)])])['reason'],'no_any_class_raw_geometry_at_IoU30')

    def test_nearby_misaligned_geometry(self):
        self.assertEqual(self.check([('a',[self.row(box=[0,0,30,10])])])['reason'],'nearby_raw_geometry_below_class_IoU50')


if __name__=='__main__':unittest.main(verbosity=2)
