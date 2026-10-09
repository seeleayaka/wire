import unittest
import numpy as np
from audit_cable_socket_controls import replay_hits

class ScalarReplayTests(unittest.TestCase):
    def fixture(self,matrix=None):
        return ([dict(id='socket',bbox_xyxy=[2,2,3,3])],
            [dict(id='socket',localization_proposal_supported=True,gates=dict(valid=True),
                inspection_to_reference_local=np.eye(3).tolist() if matrix is None else matrix)])

    def test_fragment_inventory_preserved(self):
        anchors,poses=self.fixture()
        result=replay_hits([[12],[18],[0]],5,(0,0),anchors,poses)
        self.assertEqual(result,[(1,{'socket':1}),(1,{'socket':1}),(1,{'socket':0})])

    def test_crop_original_frame_not_local_frame(self):
        anchors,poses=self.fixture()
        self.assertEqual(replay_hits([[0]],5,(2,2),anchors,poses),[(1,{'socket':1})])

    def test_unsupported_pose_cannot_add_hits(self):
        anchors,poses=self.fixture();poses[0]['gates']['valid']=False
        self.assertEqual(replay_hits([[12]],5,(0,0),anchors,poses),[(1,{'socket':0})])

    def test_mapping_horizon_rejected(self):
        anchors,poses=self.fixture([[1,0,0],[0,1,0],[0,0,0]])
        with self.assertRaises(ValueError):replay_hits([[12]],5,(0,0),anchors,poses)

if __name__=='__main__':unittest.main()
