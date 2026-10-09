import copy
import sys
import unittest
sys.path.insert(0, 'E:/PythonProject10')
from sparse_voter_nomination import nominees, actions


class NominationTests(unittest.TestCase):
    def models(self):
        box = dict(box_xyxy=[100., 100., 140., 160.], class_id=0, confidence=.8)
        return [dict(weight_sha256=w, source_sha256='source', predictions=dict(source_shape=[1000, 1000],
                     merged_predictions=[copy.deepcopy(box)] if w == 'a' else [])) for w in ['a', 'a', 'b', 'c']]

    def test_single_weight_many_views_one_nomination_not_final_cue(self):
        seed = nominees(self.models(), dict(all_predictions=[]), 'source', (1000, 1000))
        self.assertEqual(len(seed), 1)
        self.assertEqual(seed[0]['semantic_model_vote_sha256'], ['a'])
        self.assertIs(seed[0]['automatic_fault_verdict'], False)

    def test_unchanged_semantic_gate_and_action_budget(self):
        seeds = nominees(self.models(), dict(all_predictions=[]), 'source', (1000, 1000))
        self.assertEqual(actions(seeds, [[.021, .979, 0]], 5), [])
        self.assertEqual(actions(seeds, [[.001, .001, .998]], 5), [])
        self.assertEqual(actions(seeds, [[.01, .99, 0]], 0), [])
        self.assertEqual(len(actions(seeds, [[.01, .99, 0]], 1)), 1)

    def test_existing_prefix_never_renominated(self):
        models = self.models()
        existing = models[0]['predictions']['merged_predictions']
        self.assertEqual(nominees(models, dict(all_predictions=existing), 'source', (1000, 1000)), [])

    def test_source_frame_and_invalid_scores_fail_closed(self):
        with self.assertRaises(ValueError): nominees(self.models(), dict(all_predictions=[]), 'different', (1000, 1000))
        seeds = nominees(self.models(), dict(all_predictions=[]), 'source', (1000, 1000))
        with self.assertRaises(ValueError): actions(seeds, [[float('nan'), 1., 0.]], 1)
        with self.assertRaises(ValueError): actions(seeds, [], 1)

    def test_inputs_preserved_and_order_invariant(self):
        models = self.models()
        before = copy.deepcopy(models)
        a = nominees(models, dict(all_predictions=[]), 'source', (1000, 1000))
        b = nominees(list(reversed(models)), dict(all_predictions=[]), 'source', (1000, 1000))
        self.assertEqual(a, b)
        actions(a, [[.01, .99, 0]], 1)
        self.assertEqual(models, before)


if __name__ == '__main__':
    unittest.main()
