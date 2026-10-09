import unittest
import numpy as np
from socket_phenotype import fit, probability, predict, source_gate


class PhenotypeTests(unittest.TestCase):
    def data(self):
        x = np.zeros((40,320)); x[20:,0] = 1
        return x, np.array([0]*20+[1]*20)

    def test_fit_separates_and_is_deterministic(self):
        x,y=self.data(); a=fit(x,y); b=fit(x,y)
        np.testing.assert_array_equal(a['weights'],b['weights'])
        self.assertLess(probability(a,x[0]),.2); self.assertGreater(probability(a,x[-1]),.8)

    def test_ties_admit_both_and_do_not_claim_connections(self):
        x,y=self.data(); model=fit(x,y)
        r=predict(model,{0:[1.]*20,1:[1.]*20},x[0])
        self.assertIsNone(r['visual_label_candidate']); self.assertEqual(r['new_confirmed_connections'],0)
        self.assertEqual(r['confirmed_disconnections'],0)

    def test_exact_min_rank_is_one_over_21(self):
        x,y=self.data(); model=fit(x,y)
        r=predict(model,{0:[0.]*20,1:[1.]*20},x[-1])
        self.assertEqual(r['class_tail_ranks']['0'],1/21); self.assertEqual(r['visual_label_candidate'],1)

    def test_bad_inputs(self):
        x,y=self.data()
        with self.assertRaises(ValueError):fit(x,y*0)
        x[0,0]=np.nan
        with self.assertRaises(ValueError):fit(x,y)

    def test_wrong_singleton_blocks_gate(self):
        r=[{'visual_label':1,'prediction':{'visual_label_candidate':0}}]*10
        self.assertFalse(source_gate(r)['passed'])

    def test_abstention_is_not_success(self):
        r=[{'visual_label':1,'prediction':{'visual_label_candidate':None}}]*10
        self.assertEqual(source_gate(r)['singleton_coverage'],0)
        self.assertFalse(source_gate(r)['passed'])


if __name__=='__main__':unittest.main()
