import unittest,torch
from train_port_feature_adaptation import finite_loss_values
class LossContractTests(unittest.TestCase):
    def test_dictionary_loss(self):self.assertEqual(finite_loss_values({'box':torch.tensor(1.),'cls':torch.tensor(2.)}),{'box':1.,'cls':2.})
    def test_tensor_loss(self):self.assertEqual(finite_loss_values(torch.tensor([1.,2.])),{'0':1.,'1':2.})
    def test_dictionary_nan(self):
        with self.assertRaises(FloatingPointError):finite_loss_values({'box':torch.tensor(float('nan'))})
    def test_tensor_infinite(self):
        with self.assertRaises(FloatingPointError):finite_loss_values(torch.tensor([float('inf')]))
if __name__=='__main__':unittest.main()
