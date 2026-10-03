import copy
import sys
import unittest
sys.dont_write_bytecode=True
sys.path.insert(0,'E:/PythonProject10')
import numpy as np
import torch
from dense_port_probe import GRID,encode_targets,boxes_from_polygon_labels,decode,loss,DensePortHead,preprocess


class DensePortProbeTests(unittest.TestCase):
    def test_polygon_and_rectangle_conversion(self):
        a=boxes_from_polygon_labels('0 .2 .3 .4 .3 .4 .5 .2 .5',640,480)
        b=boxes_from_polygon_labels('0 .3 .4 .2 .2',640,480)
        np.testing.assert_allclose(a[0]['box'],b[0]['box'])
        self.assertEqual(a[0]['class_id'],0)
    def test_invalid_labels_fail_closed(self):
        for line in ('2 .5 .5 .1 .1','0 nan .5 .1 .1','0 .99 .5 .2 .1','0 .1 .1 .1'):
            with self.assertRaises(ValueError):boxes_from_polygon_labels(line,640,640)
    def test_empty_labels_and_finite_background_loss(self):
        t=encode_targets([],640,640)
        self.assertFalse(t['mask'].any());self.assertEqual(float(t['heat'].sum()),0)
        head=DensePortHead();out=head(torch.zeros((1,384,GRID,GRID)))
        targets={k:t[k].unsqueeze(0) for k in ('heat','reg','mask')}
        value,_=loss(out,targets);self.assertTrue(torch.isfinite(value));value.backward()
    def test_encoded_boxes_roundtrip_both_classes(self):
        boxes=[dict(class_id=0,box=[220.,260.,280.,340.]),dict(class_id=1,box=[390.,130.,430.,170.])]
        before=copy.deepcopy(boxes);t=encode_targets(boxes,640,640)
        logits=torch.full((1,2,GRID,GRID),-20.)
        logits[0][t['mask']]=20.
        rows=decode((logits,t['reg'].unsqueeze(0)),[(640,640)])[0]
        self.assertEqual(len(rows),2);self.assertEqual(boxes,before)
        for r in rows:np.testing.assert_allclose(r['box_xyxy'],boxes[r['class_id']]['box'],atol=1e-4)
    def test_same_class_same_cell_collision_deterministic(self):
        boxes=[dict(class_id=0,box=[220.,260.,280.,340.]),dict(class_id=0,box=[230.,270.,270.,330.])]
        a=encode_targets(boxes,640,640);b=encode_targets(list(reversed(boxes)),640,640)
        self.assertEqual(a['collisions'],1);self.assertTrue(torch.equal(a['reg'],b['reg']))
    def test_nonfinite_output_rejected(self):
        logits=torch.full((1,2,GRID,GRID),-20.);regs=torch.zeros((1,2,4,GRID,GRID))
        logits[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):decode((logits,regs),[(640,640)])
    def test_preprocess_and_head_contract(self):
        torch.set_num_threads(2)
        tensor=preprocess(np.zeros((640,640,3),dtype=np.uint8))
        self.assertEqual(tuple(tensor.shape),(3,448,448));self.assertTrue(torch.isfinite(tensor).all())
        out=DensePortHead()(torch.zeros((2,384,GRID,GRID)))
        self.assertEqual(tuple(out[0].shape),(2,2,GRID,GRID));self.assertEqual(tuple(out[1].shape),(2,2,4,GRID,GRID))
    def test_score_and_shape_contract(self):
        logits=torch.full((1,2,GRID,GRID),-20.);regs=torch.zeros((1,2,4,GRID,GRID))
        for score in (0,1):
            with self.assertRaises(ValueError):decode((logits,regs),[(640,640)],score=score)
        with self.assertRaises(ValueError):decode((logits,regs),[(0,640)])


if __name__=='__main__':unittest.main()
