import unittest
from audit_port_multiscale_acceptance import matches,metric

def prediction(box,cls=0):return dict(box_xyxy=box,class_id=cls)
def target(box,cls=0):return dict(box=box,class_id=cls)

class AuditTests(unittest.TestCase):
    def test_duplicate_cannot_inflate_hits(self):
        result=metric([prediction([0,0,10,10])]*2,[target([0,0,10,10])])
        self.assertEqual((result['tp'],result['unmatched']),(1,1))
    def test_class_match_required(self):
        self.assertEqual(metric([prediction([0,0,10,10],1)],[target([0,0,10,10],0)])['tp'],0)
    def test_no_targets_is_all_unmatched(self):
        self.assertEqual(metric([prediction([0,0,10,10])],[])['unmatched'],1)
    def test_distinguish_gained_and_lost_targets(self):
        labels=[target([0,0,10,10]),target([20,20,30,30])]
        old=matches([prediction([0,0,10,10])],labels)[0]
        new=matches([prediction([20,20,30,30])],labels)[0]
        self.assertEqual(new-old,{1});self.assertEqual(old-new,{0})
    def test_exact_half_iou_inclusive(self):
        self.assertEqual(metric([prediction([0,0,5,10])],[target([0,0,10,10])])['tp'],1)

if __name__=='__main__':unittest.main()
