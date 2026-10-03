import unittest
from audit_port_matching_cardinality import maximum_matching


class MatchingCardinalityTests(unittest.TestCase):
    def test_reassigns_existing_prediction_without_losing_target(self):
        # A connects both targets; B only left. The augmenting path has to
        # move A right so B can claim left, rather than dropping either.
        targets=[dict(class_id=0,box=[0,0,10,10]),dict(class_id=0,box=[4,0,14,10])]
        predictions=[dict(class_id=0,box_xyxy=[2,0,12,10]),dict(class_id=0,box_xyxy=[0,0,10,10])]
        self.assertEqual(len(maximum_matching(predictions,targets)),2)
    def test_cross_class_and_iou_boundaries(self):
        targets=[dict(class_id=0,box=[0,0,10,10])]
        self.assertEqual(maximum_matching([dict(class_id=1,box_xyxy=[0,0,10,10])],targets),{})
        self.assertEqual(maximum_matching([dict(class_id=0,box_xyxy=[9,0,19,10])],targets),{})
    def test_one_to_one_and_empty(self):
        targets=[dict(class_id=0,box=[0,0,10,10])]
        predictions=[dict(class_id=0,box_xyxy=[0,0,10,10])]*3
        self.assertEqual(len(maximum_matching(predictions,targets)),1)
        self.assertEqual(maximum_matching([],targets),{})


if __name__=='__main__':unittest.main()
