import unittest
import numpy as np
from socket_native_extent import socket_extent


class SocketNativeExtentTests(unittest.TestCase):
    def test_port_contained_pin_components_do_not_prove_outgoing_wire(self):
        raw = np.zeros((100, 150), np.uint8)
        raw[25:30, 60:70] = 1
        raw[35, 80] = 1
        rows = socket_extent(raw, np.eye(3), [50, 20, 100, 40])
        self.assertTrue(all(r['socket_pixel_support'] > 0 for r in rows))
        self.assertFalse(any(r['touches_socket_and_reaches_outgoing_context'] for r in rows))

    def test_connected_outgoing_component_keeps_even_one_pixel_touch_as_conflict(self):
        raw = np.zeros((100, 150), np.uint8)
        raw[40:81, 80] = 1
        row = socket_extent(raw, np.eye(3), [50, 20, 100, 40])[0]
        self.assertEqual(row['socket_pixel_support'], 1)
        self.assertTrue(row['touches_socket_and_reaches_outgoing_context'])

    def test_separate_far_wire_and_port_speck_cannot_be_joined(self):
        raw = np.zeros((100, 150), np.uint8)
        raw[30, 80] = 1
        raw[60:90, 80:83] = 1
        rows = socket_extent(raw, np.eye(3), [50, 20, 100, 40])
        self.assertEqual(len(rows), 2)
        self.assertFalse(any(r['touches_socket_and_reaches_outgoing_context'] for r in rows))

    def test_translation_and_reference_scale_do_not_change_decision(self):
        raw = np.zeros((100, 150), np.uint8)
        raw[30:90, 80:83] = 1
        baseline = socket_extent(raw, np.eye(3), [50, 20, 100, 40])[0]
        for scale in [0.25, 0.5, 2, 4]:
            transform = np.array([[scale, 0, -10*scale], [0, scale, -20*scale], [0, 0, 1]])
            row = socket_extent(raw, transform, np.array([50, 20, 100, 40])*scale, (10, 20))[0]
            self.assertAlmostEqual(row['outside_extent_port_short_sides'], baseline['outside_extent_port_short_sides'])
            self.assertEqual(row['socket_pixel_support'], baseline['socket_pixel_support'])

    def test_rotation_and_native_pixels_retained(self):
        raw = np.zeros((100, 150), np.uint8)
        raw[30:90, 80:83] = 1
        original = raw.copy()
        transformed = np.array([[0, -1, 100], [1, 0, 0], [0, 0, 1]])
        baseline = socket_extent(raw, np.eye(3), [50, 20, 100, 40])[0]
        rotated = socket_extent(raw, transformed, [60, 50, 80, 100])[0]
        self.assertAlmostEqual(rotated['outside_extent_port_short_sides'], baseline['outside_extent_port_short_sides'])
        self.assertTrue(np.array_equal(raw, original))

    def test_invalid_geometry_and_horizon_rejected(self):
        raw = np.ones((4, 4), np.uint8)
        for transform in [np.zeros((3, 3)), np.array([[1, 0, 0], [0, 1, 0], [1, 0, -1]])]:
            with self.assertRaises(ValueError):
                socket_extent(raw, transform, [0, 0, 2, 2])
        with self.assertRaises(ValueError):
            socket_extent(raw, np.eye(3), [0, 0, 0, 1])


if __name__ == '__main__':
    unittest.main()
