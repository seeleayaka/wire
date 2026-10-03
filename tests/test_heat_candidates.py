import unittest
import numpy as np
from tools.probe_heat_candidates import peak_boxes,choose


class HeatCandidateTests(unittest.TestCase):
    def test_normal_noise_has_no_candidates(self):
        self.assertEqual(peak_boxes(np.ones((3,4)),2,400,300),[])

    def test_two_peaks_with_weak_bridge_do_not_merge(self):
        score=np.zeros((3,7)); score[1,1]=10; score[1,5]=9; score[1,2:5]=2
        boxes=peak_boxes(score,1,700,300)
        self.assertEqual(len(boxes),2)
        self.assertEqual((boxes[0]['left'],boxes[0]['right']),(100,200))
        self.assertEqual((boxes[1]['left'],boxes[1]['right']),(500,600))

    def test_plateau_has_one_seed_and_boxes_clip(self):
        boxes=peak_boxes(np.ones((2,3))*5,2,13,11)
        self.assertEqual(len(boxes),1)
        self.assertEqual((boxes[0]['left'],boxes[0]['top'],boxes[0]['right'],boxes[0]['bottom']),(0,0,13,11))

    def test_anchor_budget_deduplication_and_input_preservation(self):
        original={'left':0,'top':0,'right':1,'bottom':1}
        new={'left':2,'top':2,'right':3,'bottom':3}
        self.assertEqual(choose([original,original],[original,new],True),[original,new])
        self.assertEqual(choose([original],[new],True),[original])
        self.assertEqual(choose([], [new],False),[])
        self.assertEqual(choose([original],[],False),[])
        self.assertEqual(original,{'left':0,'top':0,'right':1,'bottom':1})


if __name__=='__main__':
    unittest.main()
