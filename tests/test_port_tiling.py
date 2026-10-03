import unittest
import numpy as np
from inspection_agent.port_tiling import (axis_starts,tile_windows,near_artificial_edge,
    merge_tiled_ports,select_additional_tile_hint)


def pred(box,score=.8,cls=0,tile=0):
    return {'box_xyxy':box,'confidence':score,'class_id':cls,'source_tile':tile}


class TilingTests(unittest.TestCase):
    def test_coverage_and_end_anchor(self):
        windows=tile_windows(3648,2736);self.assertEqual(len(windows),12)
        coverage=np.zeros((2736,3648),np.uint8)
        for l,t,r,b in windows:coverage[t:b,l:r]=1
        self.assertTrue(coverage.all());self.assertEqual(windows[-1],(2368,1456,3648,2736))
    def test_small_image_and_invalid_inputs(self):
        self.assertEqual(tile_windows(640,480),[(0,0,640,480)])
        for bad in (0,-1,True,1.2):
            with self.assertRaises(ValueError):axis_starts(bad)
        with self.assertRaises(ValueError):axis_starts(100,20,21)
    def test_artificial_boundary_not_real_image_edge(self):
        self.assertFalse(near_artificial_edge([0,0,30,30],(0,0,1280,1280),3648,2736))
        self.assertTrue(near_artificial_edge([0,100,30,130],(960,0,2240,1280),3648,2736))
        self.assertTrue(near_artificial_edge([1250,100,1280,140],(0,0,1280,1280),3648,2736))
        self.assertFalse(near_artificial_edge([1250,1200,1280,1280],(2368,1456,3648,2736),3648,2736))
    def test_nms_keeps_geometry_and_class_separation(self):
        a=pred([10,10,30,30],.8,0,0);b=pred([11,11,31,31],.9,0,1)
        c=pred([11,11,31,31],.7,1,2)
        for order in ([a,b,c],[c,b,a]):
            out=merge_tiled_ports(order);self.assertEqual(len(out),2)
            self.assertEqual(out[0]['box_xyxy'],b['box_xyxy'])
            self.assertEqual(out[0]['support_tiles'],[0,1])
        self.assertNotIn('support_tiles',b)
    def test_invalid_predictions_rejected(self):
        for p in (pred([0,0,0,1]),pred([0,0,1,1],float('nan')),pred([0,0,1,1],cls=2)):
            with self.assertRaises(ValueError):merge_tiled_ports([p])
    def test_existing_hint_preserved_and_duplicates_removed(self):
        parents=[dict(left=0,top=0,right=100,bottom=100)]
        box=dict(left=10,top=10,right=20,bottom=20,confidence=.8,class_id=0,valid_warp_fraction=1.)
        existing=[{'box':box,'parent_index':0}]
        out=select_additional_tile_hint(parents,existing,[box],.25)
        self.assertEqual(out['existing_hints'],existing);self.assertEqual(out['tile_hints'],[])
    def test_addition_capped_one_per_image_and_order_stable(self):
        parents=[dict(left=0,top=0,right=100,bottom=100),dict(left=100,top=0,right=200,bottom=100)]
        a=dict(left=10,top=10,right=20,bottom=20,confidence=.8,class_id=0,valid_warp_fraction=1.)
        b={**a,'left':110,'right':120,'confidence':.9}
        for order in ([a,b],[b,a]):
            out=select_additional_tile_hint(parents,[],order,.25)
            self.assertEqual(len(out['tile_hints']),1);self.assertEqual(out['tile_hints'][0]['box'],b)
            self.assertEqual(out['parents'],parents)


if __name__=='__main__':unittest.main()
