import copy
import sys
import unittest
sys.dont_write_bytecode=True
sys.path.insert(0,'E:/PythonProject10')
from test_strong_model_consensus_policy import case,baseline,row
from port_support_graph_policy import append_support_graph,potential_peer_support


class SupportGraphTests(unittest.TestCase):
    def test_old_thresholds_and_symmetric_direction(self):
        for reverse in (False,True):
            a,b=case('a'),case('b');b['predictions']['merged_predictions'][0]['confidence']=.3
            for r in a['predictions']['merged_predictions']+a['predictions']['edge_kept_predictions']:r['confidence']=.8
            if reverse:a,b=b,a
            self.assertEqual(len(append_support_graph(baseline(),a,b)['strong_consensus_additions']),1)
    def test_preserve_current_prefix_metadata_and_input(self):
        old,a,b=baseline(),case('a'),case('b');before=copy.deepcopy((old,a,b))
        output=append_support_graph(old,a,b)
        self.assertEqual(output['all_predictions'][:5],old['all_predictions'])
        self.assertEqual(output['resolution_additions'],old['resolution_additions']);self.assertEqual((old,a,b),before)
    def test_same_checkpoint_not_vote(self):
        self.assertIsNotNone(append_support_graph(baseline(),case('a'),case('a'))['strong_consensus_fallback_reason'])
    def test_source_and_shape_fail_closed(self):
        for change in ('image','source_sha256','shape'):
            a,b=case('a'),case('b')
            if change=='shape':b['predictions']['source_shape']=[1,1]
            else:b[change]='different'
            r=append_support_graph(baseline(),a,b);self.assertIsNotNone(r['strong_consensus_fallback_reason'])
            self.assertEqual(r['all_predictions'],baseline()['all_predictions'])
    def test_weak_support_strict_boundary(self):
        a,b=case('a'),case('b');b['predictions']['merged_predictions'][0]['confidence']=.25
        b['predictions']['edge_kept_predictions']=[]
        self.assertEqual(append_support_graph(baseline(),a,b)['strong_consensus_additions'],[])
        self.assertEqual(potential_peer_support(b),[])
    def test_strong_boundary_and_one_view(self):
        a,b=case('a'),case('b');a['predictions']['merged_predictions'][0]['confidence']=.75
        b['predictions']['merged_predictions'][0]['confidence']=.3
        self.assertEqual(append_support_graph(baseline(),a,b)['strong_consensus_additions'],[])
        a['predictions']['merged_predictions'][0]['confidence']=.8;a['predictions']['edge_kept_predictions']=[row(1500)]
        self.assertEqual(append_support_graph(baseline(),a,b)['strong_consensus_additions'],[])
    def test_same_class_geometry_and_shared_budget(self):
        self.assertEqual(append_support_graph(baseline(),case('a'),case('b',cls=1))['strong_consensus_additions'],[])
        self.assertEqual(append_support_graph(baseline(),case('a'),case('b',x=1530))['strong_consensus_additions'],[])
        old=baseline();old['all_predictions'] += [row(700+i*60) for i in range(5)]
        self.assertEqual(append_support_graph(old,case('a'),case('b'))['strong_consensus_additions'],[])
    def test_both_classes_can_be_cues_not_fault_verdicts(self):
        for cls in (0,1):
            r=append_support_graph(baseline(),case('a',cls=cls),case('b',cls=cls))
            self.assertEqual(len(r['strong_consensus_additions']),1)
            self.assertFalse(r['strong_consensus_additions'][0]['automatic_fault_verdict'])


if __name__=='__main__':unittest.main()
