import sys
import unittest
sys.path.insert(0,'E:/PythonProject10')
import torch
from dense_pixel_port_probe import GRID,encode_targets
from dense_quality_objective import loss,iou_giou


class QualityTests(unittest.TestCase):
    def test_continuous_quality_lowers_overconfident_bad_box_not_gt_edit(self):
        target=encode_targets([dict(class_id=0,box=[200.,200.,300.,300.])],640,640)
        t={k:target[k].unsqueeze(0) for k in ('heat','reg','mask')};reg=t['reg'].clone()
        reg[:,:,2][t['mask']]+=torch.log(torch.tensor(2.))
        logits=torch.full((1,2,GRID,GRID),-20.);logits[t['mask']]=torch.log(torch.tensor(4.));logits.requires_grad_()
        value,parts=loss((logits,reg),t);value.backward()
        self.assertAlmostEqual(parts['positive_quality_mean'],.5,places=5)
        self.assertGreater(float(logits.grad[t['mask']]),0)
        self.assertTrue(torch.equal(t['reg'],target['reg'].unsqueeze(0)))
    def test_perfect_overlap_and_empty_finite_gradient(self):
        a=torch.tensor([[1.,1.,4.,4.]])
        i,g=iou_giou(a,a);self.assertEqual(float(i),1);self.assertEqual(float(g),1)
        target=encode_targets([],640,640);t={k:target[k].unsqueeze(0) for k in ('heat','reg','mask')}
        logits=torch.zeros((1,2,GRID,GRID),requires_grad=True);reg=torch.zeros((1,2,4,GRID,GRID),requires_grad=True)
        value,parts=loss((logits,reg),t);self.assertTrue(torch.isfinite(value));value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all());self.assertIsNone(parts['positive_quality_mean'])
    def test_nonfinite_fails_closed(self):
        target=encode_targets([],640,640);t={k:target[k].unsqueeze(0) for k in ('heat','reg','mask')}
        logits=torch.zeros((1,2,GRID,GRID));logits[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):loss((logits,torch.zeros((1,2,4,GRID,GRID))),t)


if __name__=='__main__':unittest.main()
