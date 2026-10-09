import unittest
from copy import deepcopy
from common_scale_evidence import common_source_qualifies, check_common_prefixes


class CommonSourceGateTests(unittest.TestCase):
    def setUp(self):
        self.totals = {name: dict(tp=tp, unmatched=4, fn=344-tp, predictions=tp+4, targets=344)
                       for name, tp in [('original', 295), ('research', 298), ('current', 300), ('trial', 301)]}
        self.rows = [dict(image=str(i)+'.JPG', lost=[], lost_original=[], lost_research=[]) for i in range(192)]

    def test_only_strict_gain_over300_can_qualify(self):
        self.assertTrue(common_source_qualifies(self.totals, 0, self.rows))
        for tp in [298, 299, 300]:
            totals = deepcopy(self.totals)
            totals['trial'].update(tp=tp, fn=344-tp, predictions=tp+4)
            self.assertFalse(common_source_qualifies(totals, 0, self.rows))

    def test_new_unmatched_normal_cue_or_old_loss_rejects(self):
        totals = deepcopy(self.totals)
        totals['trial'].update(unmatched=5, predictions=306)
        self.assertFalse(common_source_qualifies(totals, 0, self.rows))
        self.assertFalse(common_source_qualifies(self.totals, 1, self.rows))
        for key in ['lost', 'lost_original', 'lost_research']:
            rows = deepcopy(self.rows); rows[0][key] = [1]
            self.assertFalse(common_source_qualifies(self.totals, 0, rows))

    def test_partial_duplicate_or_baseline_generation_mismatch_rejects(self):
        with self.assertRaises(ValueError): common_source_qualifies(self.totals, 0, self.rows[:-1])
        rows = deepcopy(self.rows); rows[-1]['image'] = rows[0]['image']
        with self.assertRaises(ValueError): common_source_qualifies(self.totals, 0, rows)
        totals = deepcopy(self.totals); totals['current'] = totals['research']
        with self.assertRaises(ValueError): common_source_qualifies(totals, 0, self.rows)

    def test_all_generations_exact_prefix_and_shared_budget(self):
        def version(n):
            return dict(primary=[{'primary': True}], all_predictions=[{'primary': True}]+[{'cue': i} for i in range(n)])
        self.assertTrue(check_common_prefixes(version(1), version(2), version(3), version(4)))
        bad = version(4); bad['all_predictions'][2] = {'changed': True}
        with self.assertRaises(ValueError): check_common_prefixes(version(1), version(2), version(3), bad)
        with self.assertRaises(ValueError): check_common_prefixes(version(1), version(2), version(3), version(6))


if __name__ == '__main__': unittest.main(verbosity=2)
