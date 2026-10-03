import copy
import sys
import unittest
sys.dont_write_bytecode=True
sys.path.insert(0,'E:/PythonProject10')
from test_strong_model_consensus_policy import case,baseline,row
from port_support_graph_plugs_policy import append_graph_plugs


class GraphPlugTests(unittest.TestCase):
    def test_old_jack_cues_preserved_new_jacks_excluded(self):
        old=baseline();old['all_predictions'][0]['class_id']=1
        r=append_graph_plugs(old,case('a',cls=1),case('b',cls=1))
        self.assertEqual(r['strong_consensus_additions'],[]);self.assertEqual(r['all_predictions'],old['all_predictions'])
    def test_final_slot_not_consumed_by_disallowed_class(self):
        old=baseline();old['all_predictions'] += [row(700+i*60) for i in range(4)]
        a,b=case('a'),case('b')
        for c in (a,b):
            c['predictions']['merged_predictions'].insert(0,row(1800,.99,cls=1))
            c['predictions']['edge_kept_predictions'] += [row(1800,.99,tile=i,cls=1) for i in (0,1)]
        r=append_graph_plugs(old,a,b);self.assertEqual(len(r['strong_consensus_additions']),1)
        self.assertEqual(r['strong_consensus_additions'][0]['class_id'],0);self.assertEqual(len(r['all_predictions']),10)
    def test_source_identity_and_checkpoint_fail_closed(self):
        a,b=case('a'),case('a')
        self.assertIsNotNone(append_graph_plugs(baseline(),a,b)['strong_consensus_fallback_reason'])
        b=case('b');b['source_sha256']='other'
        self.assertIsNotNone(append_graph_plugs(baseline(),a,b)['strong_consensus_fallback_reason'])
    def test_old_cues_and_inputs_not_mutated(self):
        old,a,b=baseline(),case('a'),case('b');frozen=copy.deepcopy((old,a,b))
        r=append_graph_plugs(old,a,b);self.assertEqual((old,a,b),frozen)
        self.assertEqual(r['all_predictions'][:5],old['all_predictions'])
        self.assertEqual(len(r['strong_consensus_additions']),1)


if __name__=='__main__':unittest.main()
