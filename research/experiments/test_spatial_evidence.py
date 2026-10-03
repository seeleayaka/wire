import unittest
import numpy as np
from tools.audit_spatial_evidence import target_grid,rank_auc


class GridDiagnosticTests(unittest.TestCase):
    def test_boxes_are_transformed_clipped_and_do_not_wrap(self):
        mask=target_grid([[5,5,15,15],[500,500,501,501]],[10,10,30,30],(2,2))
        np.testing.assert_array_equal(mask,[[True,False],[False,False]])

    def test_auc_handles_ties_reversed_scores_and_empty_masks(self):
        mask=np.array([[True,False]])
        self.assertEqual(rank_auc(np.array([[1,0]]),mask),1)
        self.assertEqual(rank_auc(np.array([[0,1]]),mask),0)
        self.assertEqual(rank_auc(np.array([[1,1]]),mask),0.5)
        self.assertIsNone(rank_auc(np.ones((1,2)),np.zeros((1,2),bool)))


if __name__=='__main__':
    unittest.main()
