import copy
import sys
import unittest
sys.path.insert(0, 'E:/PythonProject10')
from roi_checkpoint_queries import queries
from unresolved_voter_seeds import windows


class QueryTests(unittest.TestCase):
    def fixture(self):
        seed = dict(box_xyxy=[1000, 1000, 1060, 1100], semantic_model_vote_sha256=['a', 'b'])
        models = [dict(weight_sha256=w, source_sha256='source', predictions=dict(source_shape=[2736, 3648]))
                  for w in ('a', 'a', 'b', 'c')]
        case = dict(image='arbitrary', source_sha256='source', models=models)
        control = dict(image='arbitrary', source_sha256='source', chosen_seeds=[seed],
                       ROI_evidence=[dict(seed=seed, windows=windows(seed, (2736, 3648)),
                                          missing_weight_sha256='c', views=[[], []])])
        return case, control, [seed]

    def test_all_checkpoints_once_per_action_and_exact_reuse(self):
        case, control, chosen = self.fixture()
        q = queries(case, control, chosen)
        self.assertEqual([r['weight_sha256'] for r in q], ['a', 'b', 'c'])
        self.assertEqual([r['reused_views'] for r in q], [None, None, [[], []]])

    def test_inputs_and_reused_arrays_are_not_aliased(self):
        case, control, chosen = self.fixture()
        before = copy.deepcopy((case, control, chosen))
        q = queries(case, control, chosen)
        q[-1]['reused_views'][0].append(dict(box_xyxy=[1, 2, 3, 4]))
        self.assertEqual((case, control, chosen), before)

    def test_wrong_source_or_checkpoint_is_rejected(self):
        for field in ('source', 'checkpoint', 'shape'):
            case, control, chosen = self.fixture()
            if field == 'source': control['source_sha256'] = 'other'
            if field == 'checkpoint': control['ROI_evidence'][0]['missing_weight_sha256'] = 'a'
            if field == 'shape': case['models'][0]['predictions']['source_shape'] = [3000, 3000]
            with self.assertRaises(ValueError): queries(case, control, chosen)

    def test_changed_action_cannot_be_smuggled_into_cache(self):
        case, control, chosen = self.fixture()
        bad = copy.deepcopy(chosen)
        bad[0]['box_xyxy'][0] += 1
        with self.assertRaises(ValueError): queries(case, control, bad)

    def test_no_actions_no_requests(self):
        case, control, _ = self.fixture()
        control.update(chosen_seeds=[], ROI_evidence=[])
        self.assertEqual(queries(case, control, []), [])


if __name__ == '__main__':
    unittest.main()
