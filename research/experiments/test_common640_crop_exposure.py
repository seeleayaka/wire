import unittest
from common640_crop_exposure import surviving_tiles


class ExposureTests(unittest.TestCase):
    def test_visible_inside_first_tile_and_outer_image_edge(self):
        self.assertIn(0, surviving_tiles([0, 0, 20, 20], 1000, 1000, 640, 480))
        self.assertIn(0, surviving_tiles([20, 20, 40, 40], 1000, 1000, 640, 480))

    def test_artificial_edge_not_native_image_edge(self):
        self.assertNotIn(0, surviving_tiles([620, 20, 639, 40], 1000, 1000, 640, 480))
        self.assertTrue(surviving_tiles([620, 20, 639, 40], 1000, 1000, 640, 480))

    def test_too_large_object_cannot_be_inferred_from_partial_crop(self):
        self.assertFalse(surviving_tiles([20, 20, 800, 100], 1000, 1000, 640, 480))

    def test_last_tile_and_smaller_frame_are_real_boundaries(self):
        self.assertTrue(surviving_tiles([980, 980, 1000, 1000], 1000, 1000, 640, 480))
        self.assertEqual(surviving_tiles([0, 0, 200, 100], 200, 100, 640, 480), [0])

    def test_no_survival_for_off_frame_or_nonpositive_box(self):
        for box in [[-1, 0, 20, 20], [20, 20, 20, 40], [20, 40, 30, 20]]:
            self.assertFalse(surviving_tiles(box, 1000, 1000, 640, 480))


if __name__ == '__main__': unittest.main(verbosity=2)
