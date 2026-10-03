import copy
import sys
import unittest
sys.dont_write_bytecode=True
sys.path.insert(0,'E:/PythonProject10')
from test_strong_model_consensus_policy import row,baseline
from evaluate_dense_dino_source_support import append_dense


def raw():
    return dict(source_shape=[2736,3648],merged_predictions=[row(1500,.8)],edge_kept_predictions=[row(1500,.8,i) for i in (0,1)])


class DenseSourceTests(unittest.TestCase):
    def test_independent_candidate_preserves_current_prefix(self):
        old,r=baseline(),raw();before=copy.deepcopy((old,r));result=append_dense(old,r,'new')
        self.assertEqual(len(result['dense_additions']),1);self.assertEqual(result['all_predictions'][:5],old['all_predictions'])
        self.assertEqual((old,r),before);self.assertFalse(result['dense_additions'][0]['automatic_fault_verdict'])
    def test_shared_budget_and_duplicate(self):
        old=baseline();old['all_predictions'] += [row(700+i*60) for i in range(5)]
        self.assertEqual(append_dense(old,raw(),'new')['dense_additions'],[])
        r=raw();r['merged_predictions']=[row(100,.8)];r['edge_kept_predictions']=[row(100,.8,i) for i in (0,1)]
        self.assertEqual(append_dense(baseline(),r,'new')['dense_additions'],[])
    def test_exact_score_and_two_distinct_tiles(self):
        r=raw();r['merged_predictions'][0]['confidence']=.75
        self.assertEqual(append_dense(baseline(),r,'new')['dense_additions'],[])
        r=raw();r['edge_kept_predictions']=[row(1500,.8),row(1500,.8)]
        self.assertEqual(append_dense(baseline(),r,'new')['dense_additions'],[])
    def test_invalid_source_margin_and_geometry(self):
        r=raw();r['merged_predictions'][0]['box_xyxy']=[0,300,40,340]
        self.assertEqual(append_dense(baseline(),r,'new')['dense_additions'],[])
        r=raw();r['merged_predictions'][0]['confidence']=float('nan')
        self.assertEqual(append_dense(baseline(),r,'new')['dense_additions'],[])


if __name__=='__main__':unittest.main()
