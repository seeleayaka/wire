import unittest
from paired_positive_jitter_examples import positive_jitters
from paired_semantic_geometry_examples import training_label


class PositiveTolerance(unittest.TestCase):
    def test_systematic_near_correct_boxes_and_all_GT_labels(self):
        box=[100,100,180,140];target=[dict(class_id=0,box=box)]
        rows=positive_jitters(box);self.assertEqual(len(rows),6)
        for row in rows:
            label,iou=training_label(row,target);self.assertEqual(label,1);self.assertGreaterEqual(iou,.5)
        translated=positive_jitters([200,200,280,240])
        self.assertEqual([[v+100 for v in row] for row in rows],translated)
        with self.assertRaises(ValueError):positive_jitters([1,1,1,2])

if __name__=='__main__':unittest.main()
