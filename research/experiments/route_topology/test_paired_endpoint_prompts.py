import unittest
import numpy as np
from paired_endpoint_prompts import endpoint_box

class EndpointPromptTests(unittest.TestCase):
    def test_crop_coordinate_projection(self):
        a=dict(bbox_xyxy=[20,30,40,50]);p=dict(localization_proposal_supported=True,gates=dict(ok=True),inspection_to_reference_local=np.eye(3).tolist())
        self.assertEqual(endpoint_box(a,p,[10,20,110,120]),[.2,.2,.2,.2])

    def test_no_clipping_or_unsupported_prompt(self):
        a=dict(bbox_xyxy=[0,0,20,20]);p=dict(localization_proposal_supported=True,gates=dict(ok=True),inspection_to_reference_local=np.eye(3).tolist())
        with self.assertRaises(ValueError):endpoint_box(a,p,[10,10,110,110])
        p['gates']['ok']=False
        with self.assertRaises(ValueError):endpoint_box(a,p,[0,0,100,100])

    def test_horizon_rejected(self):
        a=dict(bbox_xyxy=[0,0,10,10]);p=dict(localization_proposal_supported=True,gates=dict(ok=True),inspection_to_reference_local=[[1,0,0],[0,1,0],[1,0,1]])
        with self.assertRaises(ValueError):endpoint_box(a,p,[0,0,100,100])

if __name__=='__main__':unittest.main()
