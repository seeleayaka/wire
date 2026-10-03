import copy
import sys
import unittest
sys.path.insert(0, 'E:/PythonProject10')
from paired_port_semantic_selection import select


def proposal(x, cls=0):
    return dict(class_id=cls, confidence=.1, box_xyxy=[x, 1000, x+40, 1040],
                semantic_model_vote_sha256=['teacher', 'student'])


class SafetyTests(unittest.TestCase):
    def test_class_agnostic_duplicate_and_input_immutability(self):
        old = proposal(1000, 1)
        current = dict(primary=[old], all_predictions=[old])
        before = copy.deepcopy(current)
        candidate = proposal(1000)
        result = select(current, [candidate], [[.01, .98, .01]], 'head')
        self.assertEqual(result['paired_semantic_additions'], [])
        self.assertEqual(current, before)
        self.assertNotIn('paired_head_sha256', candidate)

    def test_shared_extra_budget_old_prefix_and_provenance(self):
        old = [proposal(500+i*60) for i in range(7)]
        current = dict(primary=old[:5], all_predictions=old)
        candidates = [proposal(1600+i*60) for i in range(8)]
        result = select(current, candidates, [[.01, .98, .01]]*8, 'fixed-head')
        self.assertEqual(len(result['all_predictions']), 10)
        self.assertEqual(result['all_predictions'][:7], old)
        self.assertEqual(len(result['paired_semantic_additions']), 3)
        self.assertTrue(all(not r['automatic_fault_verdict'] for r in result['paired_semantic_additions']))
        self.assertTrue(all(r['proposal_detector_score'] == .1 for r in result['paired_semantic_additions']))

    def test_nonfinite_inconsistent_or_out_of_range_probabilities_fail_closed(self):
        current = dict(primary=[], all_predictions=[])
        for values in ([float('nan'), 1, 0], [-.1, 1.1, 0], [0, .98, 0], [0, 1]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    select(current, [proposal(1200)], [values], 'head')
        with self.assertRaises(ValueError):
            select(current, [proposal(1200)], [], 'head')


if __name__ == '__main__':
    unittest.main()
