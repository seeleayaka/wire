import unittest
import numpy as np
from copy import deepcopy
from reference_point_seeds import select_points


class ReferencePointTests(unittest.TestCase):
    def setUp(self):
        self.raw=np.zeros((120,120),np.uint8)
        self.raw[28:33,20:101]=255
        self.anchors=[{'id':'a','bbox_xyxy':[16,26,24,34]},
                      {'id':'b','bbox_xyxy':[96,26,104,34]}]

    def test_inside_reviewed_native_pixels_and_deterministic(self):
        points,_=select_points(self.raw,self.anchors,[0,0])
        self.assertEqual(points,select_points(self.raw,self.anchors,[0,0])[0])
        for p in points:
            x,y=p['xy_crop'];self.assertTrue(self.raw[y,x]>0)
            anchor=next(a for a in self.anchors if a['id']==p['anchor_id'])
            l,t,r,b=anchor['bbox_xyxy'];self.assertTrue(l<=x<=r and t<=y<=b)

    def test_inputs_unchanged(self):
        original=self.raw.copy();anchors=deepcopy(self.anchors)
        select_points(self.raw,self.anchors,[0,0])
        np.testing.assert_array_equal(original,self.raw);self.assertEqual(anchors,self.anchors)

    def test_background_overlap_stops_not_reselected(self):
        self.raw[10,10]=255
        with self.assertRaisesRegex(ValueError,'background'):select_points(self.raw,self.anchors,[0,0])

    def test_missing_anchor_no_snapping(self):
        self.raw[:,90:]=0
        with self.assertRaisesRegex(ValueError,'does not touch'):select_points(self.raw,self.anchors,[0,0])

    def test_original_translation_invariance(self):
        points,negative=select_points(self.raw,self.anchors,[0,0])
        anchors=deepcopy(self.anchors)
        for a in anchors:a['bbox_xyxy']=[v+200 for v in a['bbox_xyxy']]
        shifted,newnegative=select_points(self.raw,anchors,[200,200])
        self.assertEqual(negative,newnegative)
        self.assertEqual([p['xy_crop'] for p in points],[p['xy_crop'] for p in shifted])

    def test_invalid_input_stops(self):
        raw=self.raw.astype(float);raw[0,0]=float('nan')
        with self.assertRaises(ValueError):select_points(raw,self.anchors,[0,0])
        with self.assertRaises(ValueError):select_points(self.raw,self.anchors,[-1,0])


if __name__=='__main__':unittest.main()
