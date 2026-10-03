import copy
import sys
import unittest
sys.path.insert(0,'E:/PythonProject10')
from dense_seed_context_policy import seeds_from_cases,context_windows,append_confirmed


def p(score=.8):return dict(class_id=0,confidence=score,box_xyxy=[1400.,1200.,1450.,1250.])
def current():return dict(primary=[],all_predictions=[])
def case(rows):return dict(source_sha256='source',predictions=dict(source_shape=[2736,3648],merged_predictions=rows))


class DenseContextPolicyTests(unittest.TestCase):
    def test_both_models_and_threshold_seed_dedup(self):
        self.assertEqual(len(seeds_from_cases(current(),[case([]),case([p()])])),1)
        self.assertEqual(len(seeds_from_cases(current(),[case([p()]),case([p()])])),1)
        self.assertEqual(seeds_from_cases(current(),[case([p(.25)]),case([])]),[])
    def test_two_strict_scores_and_prefix(self):
        seed=p();entries=[dict(seed=seed,windows=context_windows(seed,[2736,3648]),views=[[p(.9)],[p(.8)]])]
        before=current();original=copy.deepcopy(before)
        self.assertEqual(len(append_confirmed(before,entries,[2736,3648],'head')['dense_context_additions']),1)
        entries[0]['views'][1][0]['confidence']=.75
        self.assertEqual(append_confirmed(before,entries,[2736,3648],'head')['dense_context_additions'],[])
        self.assertEqual(before,original)
    def test_existing_full_budget_and_bad_provenance(self):
        old=dict(primary=[],all_predictions=[p() for _ in range(5)])
        self.assertEqual(seeds_from_cases(old,[case([p()])]),[])
        entries=[dict(seed=p(),windows=[[0,0,640,640],[0,0,960,960]],views=[[p()],[p()]])]
        with self.assertRaises(ValueError):append_confirmed(current(),entries,[2736,3648],'head')
        with self.assertRaises(ValueError):seeds_from_cases(current(),[case([]),dict(source_sha256='other',predictions=case([])['predictions'])])


if __name__=='__main__':unittest.main()
