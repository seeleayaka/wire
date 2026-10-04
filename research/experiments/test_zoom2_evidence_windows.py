import unittest
from zoom2_evidence_windows import windows


class WindowTests(unittest.TestCase):
    def test_fixed_source_extent_and_offsets(self):
        self.assertEqual(windows(dict(box_xyxy=[990, 990, 1010, 1010]), (2736, 3648)),
                         [[700, 700, 1180, 1180], [820, 820, 1300, 1300]])

    def test_image_boundary_clamps_without_resizing_window(self):
        for box in ([0, 0, 20, 20], [3628, 2716, 3648, 2736]):
            for l, t, r, b in windows(dict(box_xyxy=box), (2736, 3648)):
                self.assertEqual((r-l, b-t), (480, 480))
                self.assertTrue(0 <= l < r <= 3648 and 0 <= t < b <= 2736)

    def test_integer_translation_equivariance_away_from_edges(self):
        a = windows(dict(box_xyxy=[900, 1000, 1000, 1040]), (2736, 3648))
        b = windows(dict(box_xyxy=[917, 1029, 1017, 1069]), (2736, 3648))
        self.assertEqual(b, [[l+17, t+29, r+17, bb+29] for l, t, r, bb in a])

    def test_invalid_inputs_fail_closed(self):
        for box in ([0, 0, 0, 5], [0, 0, float('nan'), 5], [-1, 0, 3, 5]):
            with self.assertRaises(ValueError):
                windows(dict(box_xyxy=box), (2736, 3648))
        with self.assertRaises(ValueError):
            windows(dict(box_xyxy=[1, 1, 4, 5]), (479, 600))

    def test_no_mutation(self):
        row = dict(box_xyxy=[900, 1000, 1000, 1040])
        windows(row, (2736, 3648))
        self.assertEqual(row, dict(box_xyxy=[900, 1000, 1000, 1040]))

    def test_detector_coordinate_restore_is_done_once(self):
        import torch
        from ultralytics.utils.ops import scale_boxes
        network_boxes = torch.tensor([[192., 320., 384., 640.]])
        local = scale_boxes((960, 960), network_boxes, (480, 480))
        torch.testing.assert_close(local, torch.tensor([[96., 160., 192., 320.]]))
        # Result.boxes already follows this original-crop convention. The
        # caller only adds its source origin, never divides by 2 again.
        source = local + torch.tensor([[700., 700., 700., 700.]])
        torch.testing.assert_close(source, torch.tensor([[796., 860., 892., 1020.]]))


if __name__ == '__main__':
    unittest.main()
