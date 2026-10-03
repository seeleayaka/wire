import unittest
from core_port_resolution_ab_20261002 import score,ranking
def box(cls=0,confidence=.8,coordinates=None):
    return dict(class_id=cls,confidence=confidence,box_xyxy=coordinates or [0,0,10,10])
class ScoringTests(unittest.TestCase):
    def test_wrong_class_not_hit(self):
        result=score([box(1)],[dict(class_id=0,box=[0,0,10,10])])
        self.assertEqual((result['tp'],result['unmatched'],result['fn']),(0,1,1))
    def test_duplicate_predictions_only_one_hit(self):
        result=score([box(),box()],[dict(class_id=0,box=[0,0,10,10])])
        self.assertEqual((result['tp'],result['unmatched'],result['fn']),(1,1,0))
    def test_threshold_and_budget_immutable(self):
        predictions=[box(confidence=.25),box(confidence=.5),box(confidence=.9)]
        self.assertEqual(ranking(predictions,1),[predictions[2]])
        self.assertEqual(len(predictions),3)
    def test_tiny_overlap_not_precise(self):
        result=score([box(coordinates=[0,0,100,100])],[dict(class_id=0,box=[0,0,10,10])])
        self.assertEqual(result['tp'],0)
    def test_zero_labels_count_review_burden(self):
        result=score([box()],[]);self.assertEqual(result['unmatched'],1)
if __name__=='__main__':unittest.main()
