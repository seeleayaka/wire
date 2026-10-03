import unittest
import numpy as np
from paired_port_semantics import valid_boxes as square_valid_boxes
from paired_rectcontext_validity import joint_valid_boxes


class RectangularCoverage(unittest.TestCase):
    def test_subset_average_may_be_worse_than_square(self):
        mask = np.ones((1000, 1000), dtype=bool); mask[490:510] = False
        box = [400, 490, 600, 510]
        self.assertEqual(square_valid_boxes([box], mask), [0])
        self.assertEqual(joint_valid_boxes([box], mask), [])

    def test_all_valid_keeps_indices_without_relaxing_square(self):
        mask = np.ones((1000, 1000), dtype=bool)
        self.assertEqual(joint_valid_boxes([[400, 490, 600, 510], [0, 0, 40, 10]], mask), [0])


if __name__ == '__main__': unittest.main()
