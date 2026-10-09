import unittest

import numpy as np

from socket_appearance import descriptor,fit_normal,novelty,score


class SocketAppearance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(5)
        cls.features=rng.normal(.2,.02,(80,320));cls.model=fit_normal(cls.features[:40])
        cls.calibration=[score(cls.model,f) for f in cls.features[40:]]

    def test_descriptor_finite_fixed_dimensions(self):
        f=descriptor(np.zeros((50,100,3),np.uint8));self.assertEqual(f.shape,(320,));self.assertTrue(np.isfinite(f).all())

    def test_novelty_is_not_electrical_or_absence_verdict(self):
        result=novelty(self.model,self.calibration,np.ones(320))
        self.assertTrue(result['appearance_difference_candidate']);self.assertFalse(result['plug_absent_confirmed'])
        self.assertEqual(result['decision'],'insufficient_evidence')

    def test_minimum_finite_calibration_rank_not_zero(self):
        result=novelty(self.model,self.calibration,np.ones(320))
        self.assertEqual(result['normal_calibration_tail_rank'],1/41)

    def test_low_normal_score_not_normal_connection_verdict(self):
        result=novelty(self.model,self.calibration,self.model['center'])
        self.assertFalse(result['appearance_difference_candidate']);self.assertEqual(result['decision'],'insufficient_evidence')

    def test_empty_texture_regularization_finite(self):
        model=fit_normal(np.zeros((40,320)))
        self.assertEqual(score(model,np.zeros(320)),0)

    def test_invalid_patch_and_calibration_rejected(self):
        with self.assertRaises(ValueError):descriptor(np.zeros((100,100,3),np.uint8))
        with self.assertRaises(ValueError):novelty(self.model,[],np.zeros(320))


if __name__=='__main__':unittest.main()
