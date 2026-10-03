import copy
from pathlib import Path
import unittest

from tools.merge_audit import merge_variant, setup


class MergeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.tiled = setup(Path(__file__).resolve().parents[1])

    def boxes(self):
        return [{'left': x, 'top': 0, 'right': x + 100, 'bottom': 100,
                 'area': 10000, 'difference_score': 200, 'source': 'tile',
                 'source_tiles': [f'tile_{i:02d}']} for i, x in enumerate((0, 60, 120), 1)]

    def test_seed_breaks_transitive_bridge_and_preserves_observations(self):
        raw = self.boxes()
        self.assertEqual(len(merge_variant(raw, self.tiled, 'baseline')), 1)
        merged = merge_variant(raw, self.tiled, 'seed')
        self.assertEqual(len(merged), 2)
        self.assertEqual(sum(x['evidence_summary']['merged_observation_count'] for x in merged), 3)

    def test_growth_limit_prevents_excessive_group_expansion(self):
        self.assertEqual(len(merge_variant(self.boxes(), self.tiled, 'growth_1.5')), 3)

    def test_replay_policies_do_not_mutate_cached_observations(self):
        raw = self.boxes()
        before = copy.deepcopy(raw)
        for mode in ('baseline', 'overlap_only', 'seed', 'complete', 'growth_2'):
            merge_variant(raw, self.tiled, mode)
        self.assertEqual(raw, before)


if __name__ == '__main__':
    unittest.main()
