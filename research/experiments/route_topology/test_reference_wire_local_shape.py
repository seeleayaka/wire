import unittest
import numpy as np
from reference_wire_appearance_v3 import fit_reference, assess


class LocalShapeTests(unittest.TestCase):
    def masks(self):
        rgb=np.full((100,100,3),[220,30,30],np.uint8)
        horizontal=np.zeros((100,100),bool);horizontal[45:51,15:85]=True
        vertical=horizontal.T.copy()
        return rgb,horizontal,vertical

    def test_reproduces_color_only_same_color_neighbor_collision(self):
        rgb,h,v=self.masks();ref=fit_reference(rgb,h)
        self.assertEqual(assess(rgb,h,ref)[0]['state'],'reference_color_supported')
        self.assertEqual(assess(rgb,v,ref)[0]['state'],'reference_color_supported')

    def test_perpendicular_axis_is_diagnostic_mismatch(self):
        from reference_wire_local_shape import describe, compare
        _,h,v=self.masks()
        result=compare(describe(h),describe(v))
        self.assertEqual(result['axis_state'],'local_axis_differs')
        self.assertAlmostEqual(result['axis_difference_degrees'],90.)
        self.assertFalse(result['physical_identity_confirmed'])

    def test_translation_keeps_features(self):
        from reference_wire_local_shape import describe, compare
        _,h,_=self.masks();shift=np.roll(h,7,axis=0)
        self.assertEqual(compare(describe(h),describe(shift))['axis_state'],'local_axis_reference_consistent')

    def test_180_degree_axis_is_equivalent(self):
        from reference_wire_local_shape import describe, compare
        _,h,_=self.masks()
        self.assertEqual(compare(describe(h),describe(h[::-1,::-1]))['axis_difference_degrees'],0.)

    def test_empty_and_isotropic_are_unknown(self):
        from reference_wire_local_shape import describe, compare
        _,h,_=self.masks();empty=np.zeros_like(h);square=empty.copy();square[40:60,40:60]=True
        self.assertEqual(compare(describe(h),describe(empty))['axis_state'],'insufficient_local_shape')
        self.assertEqual(compare(describe(h),describe(square))['axis_state'],'insufficient_local_axis')

    def test_identical_different_wire_cannot_confirm_identity(self):
        from reference_wire_local_shape import describe, compare
        _,h,_=self.masks();result=compare(describe(h),describe(h.copy()))
        self.assertEqual(result['axis_state'],'local_axis_reference_consistent')
        self.assertFalse(result['physical_identity_confirmed'])

    def test_projective_mapping_restores_axis_without_input_mutation(self):
        from reference_wire_local_shape import describe, compare
        _,h,v=self.masks();copy=v.copy()
        H=np.array([[0.,-2.,300.],[2.,0.,-100.],[0.,0.,1.]])
        result=compare(describe(h),describe(v,H=H))
        self.assertEqual(result['axis_state'],'local_axis_reference_consistent')
        np.testing.assert_array_equal(v,copy)

    def test_degenerate_mapping_abstains(self):
        from reference_wire_local_shape import describe
        _,h,_=self.masks()
        self.assertEqual(describe(h,H=np.zeros((3,3)))['state'],'invalid_coordinate_mapping')


if __name__=='__main__':unittest.main()
