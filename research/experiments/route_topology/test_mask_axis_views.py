import unittest
import numpy as np
from mask_axis_views import axis_transform,warp_context,map_to_source


class AxisViewTests(unittest.TestCase):
    def test_horizontal_vertical_diagonal_inverse(self):
        for start,end in [((15,20),(90,20)),((20,15),(20,90)),((15,15),(80,80)),((15,80),(80,15))]:
            import cv2
            mask=np.zeros((110,110),np.uint8);cv2.line(mask,start,end,1,5)
            t=axis_transform(mask,[0,0,110,110]);self.assertEqual(t['state'],'oriented_diagnostic_view')
            points=np.array([[10.,10.],[60.,10.],[60.,20.],[10.,20.]])
            m=np.array(t['source_crop_to_oriented']);forward=points@m[:,:2].T+m[:,2]
            np.testing.assert_allclose(map_to_source(forward,t),points,atol=1e-10)

    def test_isotropic_mask_abstains(self):
        mask=np.zeros((60,60));mask[10:30,10:30]=1
        self.assertEqual(axis_transform(mask,[0,0,60,60])['state'],'orientation_ambiguous')

    def test_empty_single_pixel_abstains(self):
        mask=np.zeros((60,60))
        for count in [0,1]:
            if count:mask[10,10]=1
            self.assertEqual(axis_transform(mask,[0,0,60,60])['state'],'insufficient_mask_pixels')

    def test_whole_mask_crop_required(self):
        mask=np.zeros((60,60));mask[10:40,10:15]=1
        with self.assertRaises(ValueError):axis_transform(mask,[0,0,20,20])

    def test_original_crop_translation(self):
        mask=np.zeros((80,100));mask[20:50,30:35]=1
        t=axis_transform(mask,[25,15,40,55]);m=np.array(t['source_crop_to_oriented'])
        points=np.array([[30.,25.],[34.,25.],[34.,40.],[30.,40.]])
        shifted=points-[25,15];forward=shifted@m[:,:2].T+m[:,2]
        np.testing.assert_allclose(map_to_source(forward,t),points,atol=1e-10)

    def test_frame_corners_not_cut_off(self):
        mask=np.eye(50);t=axis_transform(mask,[0,0,50,50]);m=np.array(t['source_crop_to_oriented'])
        p=np.array([[0,0],[49,0],[49,49],[0,49]])@m[:,:2].T+m[:,2]
        self.assertTrue(np.all(p>=1.999999))
        self.assertTrue(np.all(p[:,0]<t['output_size'][0]-1))
        self.assertTrue(np.all(p[:,1]<t['output_size'][1]-1))

    def test_warp_does_not_mutate_inputs(self):
        mask=np.zeros((80,100));mask[20:50,30:35]=1;image=np.zeros((80,100,3),np.uint8)
        before=image.copy();t=axis_transform(mask,[25,15,40,55]);out=warp_context(image,[25,15,40,55],t)
        np.testing.assert_array_equal(before,image)
        self.assertEqual(out.shape[:2],tuple(reversed(t['output_size'])))
        self.assertIsNone(t['automatic_wire_identity']);self.assertEqual(t['observer_count'],1)

    def test_seeded_arbitrary_positions_and_sizes_roundtrip(self):
        import cv2
        rng=np.random.default_rng(20261005)
        for _ in range(80):
            w=int(rng.integers(80,180));h=int(rng.integers(80,180));mask=np.zeros((h,w),np.uint8)
            start=(int(rng.integers(10,w-10)),int(rng.integers(10,h-10)))
            end=(int(rng.integers(10,w-10)),int(rng.integers(10,h-10)))
            cv2.line(mask,start,end,1,2);t=axis_transform(mask,[0,0,w,h])
            if t['state']!='oriented_diagnostic_view':continue
            m=np.array(t['source_crop_to_oriented']);p=np.array([[10.,10.],[20.,10.],[20.,20.],[10.,20.]])
            np.testing.assert_allclose(map_to_source(p@m[:,:2].T+m[:,2],t),p,atol=1e-10)


if __name__=='__main__':unittest.main()
