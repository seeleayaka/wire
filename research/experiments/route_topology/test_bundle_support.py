from copy import deepcopy
import unittest

import numpy as np

from bundle_support import inspect_support
import test_visible_lead_scope


class BundleSupport(unittest.TestCase):
    def setUp(self):
        fixture=test_visible_lead_scope.VisibleLeadScope();fixture.setUp();self.binding=fixture.binding;self.scope=fixture.scope
        self.raw=np.zeros((120,120),np.uint8);self.raw[30,20:101]=255

    def inspect(self,raw=None,**kwargs):
        return inspect_support(self.raw if raw is None else raw,.9,'fixture',self.scope,self.binding,**kwargs)

    def test_raw_two_anchor_support_not_electrical_success(self):
        result=self.inspect()
        self.assertTrue(result['component_has_both_anchor_support'])
        self.assertFalse(result['attachment_confirmed'])
        self.assertEqual(result['decision'],'insufficient_evidence')

    def test_no_gap_fill(self):
        raw=self.raw.copy();raw[30,60]=0
        result=self.inspect(raw)
        self.assertEqual(result['raw_foreground_component_count'],2)
        self.assertFalse(result['component_has_both_anchor_support'])

    def test_branch_and_extra_component_retained_not_promoted(self):
        raw=self.raw.copy();raw[30:70,60]=255;raw[80,80]=255
        result=self.inspect(raw)
        self.assertTrue(result['component_has_both_anchor_support'])
        self.assertEqual(result['raw_foreground_component_count'],2)
        self.assertGreater(result['whole_mask_geometry']['branch_pixel_count'],0)
        self.assertFalse(result['qualified_single_wire_path'])

    def test_original_pixels_unchanged(self):
        raw=self.raw.copy();before=raw.copy();scope=deepcopy(self.scope)
        self.inspect(raw)
        np.testing.assert_array_equal(before,raw);self.assertEqual(scope,self.scope)

    def test_nearest_anchor_not_used(self):
        raw=self.raw.copy();raw[30,20:25]=0
        self.assertFalse(self.inspect(raw)['component_has_both_anchor_support'])

    def test_empty_is_insufficient(self):
        result=self.inspect(np.zeros_like(self.raw))
        self.assertEqual(result['components'],[]);self.assertEqual(result['decision'],'insufficient_evidence')

    def test_exact_crop_translation_and_mapping(self):
        raw=np.zeros((90,100),np.uint8);raw[20,10:91]=255
        self.assertTrue(self.inspect(raw,translation=(15,12),
            matrix=np.array([[1,0,-5],[0,1,-2],[0,0,1]]))['component_has_both_anchor_support'])


if __name__=='__main__':unittest.main()
