import unittest
import torch
from tiny_paired_semantics import TinyPairedHead,fit


class TinyPairedContract(unittest.TestCase):
    def test_fixed_small_architecture_and_deterministic_eval(self):
        head=TinyPairedHead().eval()
        self.assertEqual(sum(p.numel() for p in head.parameters()),98371)
        image_features=torch.ones((2,6144))
        self.assertEqual(head(image_features).shape,(2,3))
        self.assertTrue(torch.equal(head(image_features),head(image_features)))

    def test_bad_training_data_rejected_before_optimizing(self):
        for features,labels in ((torch.ones(2,4),torch.tensor([0])),
            (torch.ones(3,4),torch.tensor([0,0,1])),
            (torch.full((3,4),float('nan')),torch.tensor([0,1,2])),
            (torch.ones(3,4),torch.tensor([0,1,3])),
            (torch.ones(3,4),torch.tensor([0.,1.,2.]))):
            with self.assertRaises(ValueError):fit(features,labels)

    def test_seeded_fixed_fit_reproducible_and_does_not_modify_inputs(self):
        torch.set_num_threads(2)
        features=torch.eye(3).repeat(4,1);labels=torch.tensor([0,1,2]*4)
        original=features.clone();targets=labels.clone()
        first=fit(features,labels);second=fit(features,labels)
        self.assertTrue(all(torch.equal(v,second.state_dict()[k]) for k,v in first.state_dict().items()))
        self.assertTrue(torch.equal(features,original) and torch.equal(labels,targets))
        self.assertFalse(first.training)
        self.assertTrue(all(not p.requires_grad for p in first.parameters()))


if __name__=='__main__':unittest.main()
