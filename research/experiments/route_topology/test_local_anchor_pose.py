from copy import deepcopy
import unittest

import cv2
import numpy as np

from local_anchor_pose import localize_anchor,project_points


class LocalAnchorPose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(20261005)
        gray=rng.integers(0,256,(800,800),dtype=np.uint8)
        gray=cv2.GaussianBlur(gray,(3,3),.6)
        cls.image=np.repeat(gray[:,:,None],3,axis=2)
        cls.anchor={'id':'constructed_anchor','kind':'wire_entry_socket','bbox_xyxy':[350,350,400,400]}

    def test_identity_background_is_localization_not_connection(self):
        r=localize_anchor(self.image,self.image,self.anchor,np.eye(3))
        self.assertTrue(r['reliable_localization'])
        self.assertFalse(r['confirmed']);self.assertFalse(r['plug_seating_assessed'])
        self.assertEqual(r['electrical_continuity'],'not_assessed')

    def test_translation_direction(self):
        forward=np.array([[1,0,7],[0,1,5],[0,0,1]],float)
        inspection=cv2.warpPerspective(self.image,forward,(800,800))
        r=localize_anchor(self.image,inspection,self.anchor,np.linalg.inv(forward))
        self.assertTrue(r['reliable_localization'])
        self.assertLess(r['local_global_corner_disagreement_px'],1)
        self.assertAlmostEqual(r['inspection_anchor_polygon_xy'][0][0],357,delta=1)

    def test_anchor_interior_changes_cannot_certify_seating(self):
        inspection=self.image.copy();inspection[350:400,350:400]=0
        r=localize_anchor(self.image,inspection,self.anchor,np.eye(3))
        self.assertTrue(r['reliable_localization'])
        self.assertTrue(r['anchor_interior_feature_matches_excluded'])
        self.assertFalse(r['plug_seating_assessed'])

    def test_blank_context_declines(self):
        blank=np.zeros_like(self.image)
        self.assertFalse(localize_anchor(blank,blank,self.anchor,np.eye(3))['reliable_localization'])

    def test_outside_context_not_clipped(self):
        anchor=deepcopy(self.anchor);anchor['bbox_xyxy']=[0,0,50,50]
        self.assertEqual(localize_anchor(self.image,self.image,anchor,np.eye(3))['reason'],
                         'reference_context_outside_image')

    def test_invalid_projection_declines(self):
        with self.assertRaises(ValueError):project_points([[0,0]],np.zeros((3,3)))


if __name__=='__main__':unittest.main()
