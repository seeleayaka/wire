import unittest
import numpy as np
import torch
from paired_rectcontext_features import crop_signature, crop_tensor, embeddings
from paired_shape_features import crop_signature as square_signature


class RectangularContext(unittest.TestCase):
    def test_true_short_axis_changes_input(self):
        wide, narrow = [40, 80, 120, 120], [40, 95, 120, 105]
        self.assertEqual(square_signature(wide, 1.5), square_signature(narrow, 1.5))
        self.assertNotEqual(crop_signature(wide, 1.5), crop_signature(narrow, 1.5))
        image = np.arange(200 * 200 * 3, dtype=np.uint8).reshape(200, 200, 3)
        self.assertFalse(torch.equal(crop_tensor(image, crop_signature(wide, 1.5)), crop_tensor(image, crop_signature(narrow, 1.5))))
        with self.assertRaises(ValueError): crop_signature([1, 1, 1, 2], 1.5)

    def test_duplicate_actual_crops_forward_once(self):
        class Encoder:
            forwarded = 0
            def forward_features(self, batch):
                self.forwarded += len(batch)
                return dict(x_norm_clstoken=torch.ones(len(batch), 384), x_norm_patchtokens=torch.ones(len(batch), 256, 384))
        model = Encoder(); audit = {}; image = np.zeros((200, 200, 3), dtype=np.uint8)
        value = embeddings(model, image, [[40, 80, 120, 120]] * 2, audit=audit)
        self.assertEqual(value.shape, (2, 1536)); self.assertEqual(model.forwarded, 2)
        self.assertEqual(audit['requested_crops'], 4); self.assertEqual(audit['unique_crops'], 2)
        self.assertTrue(torch.equal(value[0], value[1]))


if __name__ == '__main__': unittest.main()
