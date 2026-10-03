import copy
import sys
import unittest
sys.dont_write_bytecode=True
sys.path.insert(0,'E:/PythonProject10')
import numpy as np
import torch
from dense_pixel_port_probe import GRID,INPUT,DensePixelPortHead,encode_targets,decode,loss


class DensePixelProbeTests(unittest.TestCase):
    def test_high_resolution_head_contract_and_pixel_gradient(self):
        torch.set_num_threads(2);head=DensePixelPortHead()
        features=dict(semantic=torch.zeros((1,384,32,32)),rgb=torch.randn((1,3,INPUT,INPUT)))
        out=head(features);self.assertEqual(tuple(out[0].shape),(1,2,56,56))
        t=encode_targets([dict(class_id=0,box=[220.,260.,280.,340.])],640,640)
        value,_=loss(out,{k:t[k].unsqueeze(0) for k in ('heat','reg','mask')});self.assertTrue(torch.isfinite(value));value.backward()
        self.assertGreater(float(head.pixel[0].weight.grad.abs().sum()),0)
    def test_roundtrip_unchanged_geometry(self):
        boxes=[dict(class_id=0,box=[220.,260.,280.,340.]),dict(class_id=1,box=[390.,130.,430.,170.])]
        before=copy.deepcopy(boxes);t=encode_targets(boxes,640,640);logits=torch.full((1,2,GRID,GRID),-20.)
        logits[0][t['mask']]=20.;rows=decode((logits,t['reg'].unsqueeze(0)),[(640,640)])[0]
        for r in rows:np.testing.assert_allclose(r['box_xyxy'],boxes[r['class_id']]['box'],atol=1e-4)
        self.assertEqual(boxes,before);self.assertEqual(len(rows),2)
    def test_empty_and_invalid_contracts(self):
        self.assertFalse(encode_targets([],640,640)['mask'].any())
        head=DensePixelPortHead()
        with self.assertRaises(ValueError):head(dict(semantic=torch.zeros((1,384,32,32)),rgb=torch.zeros((1,3,224,224))))
        logits=torch.full((1,2,GRID,GRID),-20.);regs=torch.zeros((1,2,4,GRID,GRID));logits[0,0,0,0]=float('inf')
        with self.assertRaises(ValueError):decode((logits,regs),[(640,640)])


if __name__=='__main__':unittest.main()
