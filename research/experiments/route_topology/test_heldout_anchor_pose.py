from copy import deepcopy
import unittest

import numpy as np

from heldout_anchor_pose import fit_with_holdout


class HeldoutAnchorPose(unittest.TestCase):
    def setUp(self):
        self.dst=np.array([(x,y) for x in range(200,601,20) for y in range(200,601,20)],np.float32)
        self.anchor={'id':'fixture','bbox_xyxy':[350,350,400,400]}
        self.context=[200,200,601,601]

    def fit(self,src=None,dst=None,global_matrix=None,anchor=None):
        return fit_with_holdout(self.dst if src is None else src,self.dst if dst is None else dst,
            self.anchor if anchor is None else anchor,self.context,np.eye(3) if global_matrix is None else global_matrix)

    def test_identity_heldout_not_connection(self):
        result=self.fit();self.assertTrue(result['localization_proposal_supported'])
        self.assertFalse(result['heldout_points_used_to_fit']);self.assertFalse(result['identity_verified'])

    def test_report_json_native_booleans(self):
        import json
        result=self.fit();json.dumps(result)
        self.assertTrue(all(type(v) is bool for v in result['gates'].values()))

    def test_large_global_disagreement_explicit_not_old_gate_override(self):
        result=self.fit(self.dst+[10,8])
        self.assertTrue(result['localization_proposal_supported'])
        self.assertGreater(result['local_global_corner_disagreement_px'],10)
        self.assertFalse(result['global_agreement_gate_replaced'])

    def test_corrupted_check_points_cannot_train_away_error(self):
        import hashlib
        heldout=np.array([int.from_bytes(hashlib.sha256(f'{round(float(x))},{round(float(y))}'.encode('ascii')).digest()[:4],
            'big')%3==0 for x,y in self.dst])
        source=self.dst.copy();source[heldout]+=[20,30]
        result=self.fit(source);self.assertFalse(result['localization_proposal_supported'])
        self.assertFalse(result['gates']['heldout_reprojection'])

    def test_target_extrapolation_rejected(self):
        anchor=deepcopy(self.anchor);anchor['bbox_xyxy']=[630,350,670,400]
        result=self.fit(anchor=anchor);self.assertFalse(result['gates']['anchor_in_training_hull'])

    def test_too_few_not_accepted(self):
        result=self.fit(self.dst[:4],self.dst[:4]);self.assertFalse(result['localization_proposal_supported'])

    def test_duplicates_same_pixel_cannot_cross_split(self):
        result=self.fit(np.tile([[350,350]],(50,1)),np.tile([[350,350]],(50,1)))
        self.assertFalse(result['localization_proposal_supported'])

    def test_nonfinite_inputs_rejected(self):
        source=self.dst.copy();source[0,0]=np.nan
        with self.assertRaises(ValueError):self.fit(source)


if __name__=='__main__':unittest.main()
