import copy
import unittest
from paired_median_proposals import proposals


def model(weight, box, confidence=.6, source='s'):
    return dict(source_sha256=source, weight_sha256=weight,
        predictions=dict(source_shape=[200, 200], merged_predictions=[dict(
            class_id=0, box_xyxy=box, confidence=confidence)]))


class MedianProposals(unittest.TestCase):
    def test_unique_checkpoint_views_only_vote_once(self):
        a = model('a', [30, 40, 70, 60], .9)
        b = model('a', [32, 40, 72, 60], .8)
        self.assertEqual(proposals(a, [a, b]), [])
        c = model('c', [34, 40, 74, 60], .6)
        rows = proposals(a, [a, b, c])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['box_xyxy'], [32., 40., 72., 60.])
        self.assertEqual(rows[0]['median_unique_checkpoint_count'], 2)
        self.assertEqual(rows[0]['median_support_boxes']['a'], a['predictions']['merged_predictions'][0]['box_xyxy'])

    def test_order_and_translation_invariance_and_no_mutation(self):
        a, b = model('a', [30, 40, 70, 60]), model('b', [34, 40, 74, 60])
        before = copy.deepcopy([a, b])
        self.assertEqual(proposals(a, [a, b]), proposals(a, [b, a]))
        self.assertEqual([a, b], before)
        for item in (a, b):
            item['predictions']['merged_predictions'][0]['box_xyxy'] = [v + 20 for v in item['predictions']['merged_predictions'][0]['box_xyxy']]
        self.assertEqual(proposals(a, [a, b])[0]['box_xyxy'], [52., 60., 92., 80.])

    def test_source_mismatch_rejected(self):
        a = model('a', [30, 40, 70, 60])
        with self.assertRaises(ValueError):
            proposals(a, [a, model('b', [30, 40, 70, 60], source='other')])


if __name__ == '__main__': unittest.main()
