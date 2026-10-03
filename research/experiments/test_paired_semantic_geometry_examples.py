import unittest
from paired_semantic_geometry_examples import perturbations,training_label


class GeometryTests(unittest.TestCase):
    def test_six_uniform_examples_no_coordinate_lookup(self):
        box=[100,100,140,160];rows=perturbations(box)
        self.assertEqual(len(rows),6)
        labels=[training_label(row,[dict(class_id=0,box=box)])[0] for row in rows]
        self.assertEqual(labels,[0]*6)
        with self.assertRaises(ValueError):perturbations([1,2,1,3])

    def test_neighboring_GT_must_not_be_a_blind_negative(self):
        targets=[dict(class_id=0,box=[100,100,140,140]),dict(class_id=1,box=[130,100,170,140])]
        shifted=perturbations(targets[0]['box'])[1]
        label,iou=training_label(shifted,targets)
        self.assertEqual(label,2);self.assertEqual(iou,1.)
        self.assertEqual(training_label([0,0,1,1],[]),(0,0.))


if __name__=='__main__':unittest.main()
