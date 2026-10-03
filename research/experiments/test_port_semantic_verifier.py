import unittest
import numpy as np
import torch
from port_semantic_verifier import crop_tensor,coverage,context_box,precision_gate
class SemanticTests(unittest.TestCase):
    def test_context_excludes_hidden_positive_outside_tight_box(self):
        self.assertEqual(coverage([40,40,60,60],[70,40,80,50]),0.)
        self.assertGreater(coverage(context_box([40,40,60,60],3.),[70,40,80,50]),0.)
    def test_box_geometry_and_square_input(self):
        self.assertEqual(tuple(crop_tensor(np.zeros((100,120,3),dtype=np.uint8),[5,10,25,50],1.5).shape),(3,224,224))
    def test_edge_padding_is_finite(self):
        self.assertTrue(torch.isfinite(crop_tensor(np.zeros((100,120,3),dtype=np.uint8),[0,0,10,20],3)).all())
    def test_invalid_and_external_boxes_rejected(self):
        for box in ([5,5,1,1],[float('nan'),5,10,10],[200,200,220,220]):
            with self.assertRaises(ValueError):crop_tensor(np.zeros((100,120,3),dtype=np.uint8),box,1.5)
    def test_coverage_catches_small_port_inside_big_crop(self):
        self.assertEqual(coverage([0,0,100,100],[5,5,10,10]),1.)
        self.assertEqual(coverage([0,0,10,10],[20,20,30,30]),0.)
    def test_empty_acceptance_does_not_pass(self):
        result=precision_gate(torch.tensor([[.5,.3,.2]]),torch.tensor([1]))
        self.assertFalse(result['qualifies']);self.assertEqual(result['precision'],0.)
    def test_wrong_port_class_counts_as_error(self):
        result=precision_gate(torch.tensor([[.001,.001,.998]]),torch.tensor([1]))
        self.assertEqual(result['fp'],1);self.assertFalse(result['qualifies'])
    def test_fixed_gate_accepts_correct_port(self):
        result=precision_gate(torch.tensor([[.001,.998,.001]]),torch.tensor([1]))
        self.assertTrue(result['qualifies'])
if __name__=='__main__':unittest.main()
