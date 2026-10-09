import unittest
from copy import deepcopy
from common_scale_evidence import audit_common_views, validate_fallback_prerequisites, tile_windows


class CommonEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.roles = dict(teacher='a'*64, student='b'*64, feature='c'*64)
        self.source = 'd'*64
        self.shape = [2736, 3648]
        grids = [[[0, 0, 3648, 2736]]] + [
            [list(w) for w in tile_windows(3648, 2736, size, stride)]
            for size, stride in [(1280, 960), (960, 720), (640, 480)]]
        self.views = [dict(weight_sha256=d, source_sha256=self.source,
                           predictions=dict(source_shape=self.shape, windows=g,
                                            raw_predictions=0, edge_rejected=0,
                                            edge_kept_predictions=[], merged_predictions=[]))
                      for d in self.roles.values() for g in grids]

    def check(self, views=None):
        return audit_common_views(self.views if views is None else views, self.roles, self.source, self.shape)

    def test_actual_empty_views_are12_recipes_not12_checkpoint_votes(self):
        before = deepcopy(self.views)
        self.assertEqual(self.check(), 12)
        self.assertEqual(before, self.views)
        self.assertEqual(len({v['weight_sha256'] for v in self.views}), 3)

    def test_missing_or_duplicate_common_recipe(self):
        for views in [self.views[:-1], self.views+[deepcopy(self.views[-1])]]:
            with self.assertRaises(ValueError): self.check(views)

    def test_repeated_teacher_cannot_supply_student_vote(self):
        views = deepcopy(self.views)
        views[7]['weight_sha256'] = self.roles['teacher']
        with self.assertRaises(ValueError): self.check(views)

    def test_placeholder_and_wrong_frame_fail(self):
        for key, value in [('windows', []), ('source_shape', [3648, 2736])]:
            views = deepcopy(self.views)
            views[3]['predictions'][key] = value
            with self.assertRaises(ValueError): self.check(views)

    def test_count_conservation(self):
        for value in [1, True, -1]:
            views = deepcopy(self.views)
            views[3]['predictions']['raw_predictions'] = value
            with self.assertRaises(ValueError): self.check(views)

    def test_cut_edges_wrong_tile_and_bad_class_are_not_evidence(self):
        for row in [
            dict(box_xyxy=[481, 20, 500, 40], source_tile=1, class_id=0, confidence=.9),
            dict(box_xyxy=[20, 20, 40, 40], source_tile=1, class_id=0, confidence=.9),
            dict(box_xyxy=[20, 20, 40, 40], source_tile=True, class_id=0, confidence=.9),
            dict(box_xyxy=[20, 20, 40, 40], source_tile=0, class_id=True, confidence=.9),
        ]:
            views = deepcopy(self.views)
            p = views[3]['predictions']
            p.update(raw_predictions=1, edge_kept_predictions=[row], merged_predictions=[])
            with self.assertRaises(ValueError): self.check(views)

    def test_real_kept_detection_merge_replays_without_changing_input(self):
        views = deepcopy(self.views)
        row = dict(box_xyxy=[20, 20, 40, 40], source_tile=0, class_id=0, confidence=.9)
        p = views[3]['predictions']
        p.update(raw_predictions=1, edge_kept_predictions=[row], merged_predictions=[dict(row, support_tiles=[0])])
        self.assertEqual(self.check(views), 12)

    def test_live_incomplete_or_successful_development_blocks_fallback(self):
        source = dict(qualifies=True, normal_cues=0, summary={'trial': {'tp': 300, 'unmatched': 4}})
        audit = dict(status='pass', candidate_source_qualifies=True)
        stages = {'inner': {'count': 48, 'qualifies': False}}
        done = dict(status='rejected', failed_stage='inner', stages=stages)
        da = dict(status='pass', candidate_development_qualifies=False, stages=stages)
        self.assertTrue(validate_fallback_prerequisites(source, audit, done, da))
        for status in ['running', 'failed', 'complete', 'accepted']:
            with self.assertRaises(ValueError):
                validate_fallback_prerequisites(source, audit, dict(done, status=status), da)
        with self.assertRaises(ValueError):
            validate_fallback_prerequisites(source, audit, done, dict(da, status='pending'))

    def test_failed_outer_requires_completed_passing_inner(self):
        source = dict(qualifies=True, normal_cues=0, summary={'trial': {'tp': 300, 'unmatched': 4}})
        audit = dict(status='pass', candidate_source_qualifies=True)
        stages = {'inner': {'count': 48, 'qualifies': True}, 'outer': {'count': 30, 'qualifies': False}}
        dev = dict(status='rejected', failed_stage='outer', stages=stages)
        da = dict(status='pass', candidate_development_qualifies=False, stages=stages)
        self.assertTrue(validate_fallback_prerequisites(source, audit, dev, da))
        for change in [dict(stages={'outer': stages['outer']}), dict(failed_stage=None),
                       dict(stages={'inner': dict(count=48, qualifies=False), 'outer': stages['outer']})]:
            with self.assertRaises(ValueError):
                validate_fallback_prerequisites(source, audit, dict(dev, **change), da)


if __name__ == '__main__': unittest.main(verbosity=2)
