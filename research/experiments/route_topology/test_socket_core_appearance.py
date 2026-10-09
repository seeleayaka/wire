import unittest

import numpy as np

from socket_core_appearance import learn_core,descriptor


class SocketCoreAppearance(unittest.TestCase):
    def setUp(self):
        self.patch=np.full((50,100,3),180,np.uint8);self.patch[15:35,25:75]=15
        self.patches=np.repeat(self.patch[None],40,axis=0)

    def test_core_learned_from_normal_only(self):
        core,report=learn_core(self.patches)
        self.assertGreaterEqual(report['active_pixels'],50);self.assertFalse(report['plug_semantics_verified'])
        self.assertEqual(descriptor(self.patch,core).shape,(320,))

    def test_outside_route_color_does_not_enter_features(self):
        core,_=learn_core(self.patches);changed=self.patch.copy();changed[:10]=[255,10,10]
        np.testing.assert_array_equal(descriptor(self.patch,core),descriptor(changed,core))

    def test_inside_appearance_changes_remain_visible(self):
        core,_=learn_core(self.patches);changed=self.patch.copy();changed[20:30,40:60]=220
        self.assertGreater(np.linalg.norm(descriptor(self.patch,core)-descriptor(changed,core)),.1)

    def test_dark_region_elsewhere_not_nearest_snapped_to_center(self):
        patches=np.full_like(self.patches,180);patches[:,5:15,5:20]=15
        with self.assertRaises(ValueError):learn_core(patches)

    def test_insufficient_core_cannot_be_widened(self):
        patches=np.full_like(self.patches,180);patches[:,24:27,49:52]=15
        with self.assertRaises(ValueError):learn_core(patches)

    def test_no_input_mutation(self):
        before=self.patches.copy();learn_core(self.patches);np.testing.assert_array_equal(before,self.patches)


if __name__=='__main__':unittest.main()
