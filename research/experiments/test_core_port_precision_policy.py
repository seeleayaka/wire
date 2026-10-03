import copy
import unittest
from core_port_precision_policy import select

def packet(conf=.4, support=.3, box=None, second_class=0):
    box = box or [1000,100,1100,200]
    return dict(source_shape=[1280,2240], windows=[[0,0,1280,1280],[960,0,2240,1280]],
        merged_predictions=[dict(box_xyxy=box,class_id=0,confidence=conf,source_tile=0,support_tiles=[0,1])],
        edge_kept_predictions=[dict(box_xyxy=box,class_id=0,confidence=conf,source_tile=0),
                               dict(box_xyxy=box,class_id=second_class,confidence=support,source_tile=1)])

class Policies(unittest.TestCase):
    def test_repeated_weak_kept(self):
        self.assertEqual(len(select(packet(), 'weak_overlap_consensus')),1)
    def test_low_support_rejected(self):
        self.assertEqual(select(packet(support=.1),'weak_overlap_consensus'),[])
    def test_wrong_class_rejected(self):
        self.assertEqual(select(packet(second_class=1),'weak_overlap_consensus'),[])
    def test_single_view_not_penalized(self):
        self.assertEqual(len(select(packet(box=[100,100,200,200],support=.1),'weak_overlap_consensus')),1)
    def test_high_score_not_penalized(self):
        self.assertEqual(len(select(packet(conf=.6,support=.1),'weak_overlap_consensus')),1)
    def test_strict_boundary(self):
        self.assertEqual(select(packet(conf=.5),'high_score'),[])
    def test_immutable(self):
        p=packet(); original=copy.deepcopy(p)
        for mode in ('bounded','high_score','weak_overlap_consensus'):select(p,mode)
        self.assertEqual(p,original)
    def test_invalid_mode(self):
        with self.assertRaises(ValueError):select(packet(),'bad')
    def test_complete_frame_all_four_edges(self):
        for box in ([0,100,80,200],[100,0,200,80],[2160,100,2240,200],[100,1200,200,1280]):
            self.assertEqual(select(packet(conf=.9,box=box),'high_score_complete_frame'),[])
    def test_complete_frame_keeps_interior(self):
        self.assertEqual(len(select(packet(conf=.9),'high_score_complete_frame')),1)
    def test_complete_frame_margin_strict(self):
        self.assertEqual(select(packet(conf=.9,box=[16,100,80,200]),'high_score_complete_frame'),[])

if __name__=='__main__':unittest.main()
