import copy
import sys
import unittest
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0, 'E:/PythonProject10')
sys.path.insert(0, str(Path(__file__).parent))
from strong_model_consensus_policy import append_strong_consensus


def row(x, score=.95, tile=0, cls=0):
    return dict(box_xyxy=[x, 300, x+40, 340], confidence=score, class_id=cls, source_tile=tile)


def case(weight, x=1500, cls=0):
    return dict(image='anything.JPG', source_sha256='source', weight_sha256=weight, zoom_evidence=[],
                predictions=dict(source_shape=[2736, 3648], merged_predictions=[row(x, cls=cls)],
                                 edge_kept_predictions=[row(x, tile=i, cls=cls) for i in (0,1)]))


def baseline():
    rows=[row(100+i*60) for i in range(5)]
    return dict(primary=rows, all_predictions=copy.deepcopy(rows), resolution_additions=[dict(old='preserved')])


class StrongConsensusTests(unittest.TestCase):
    def test_exact_inputs_and_whole_current_prefix_preserved(self):
        old, a, b = baseline(), case('a'), case('b')
        frozen=copy.deepcopy((old,a,b))
        result=append_strong_consensus(old,a,b)
        self.assertEqual(len(result['strong_consensus_additions']),1)
        self.assertEqual(result['all_predictions'][:5],old['all_predictions'])
        self.assertEqual(result['resolution_additions'],old['resolution_additions'])
        self.assertEqual((old,a,b),frozen)
        self.assertFalse(result['strong_consensus_additions'][0]['automatic_fault_verdict'])
    def test_same_checkpoint_not_second_vote(self):
        self.assertIsNotNone(append_strong_consensus(baseline(),case('a'),case('a'))['strong_consensus_fallback_reason'])
    def test_identity_or_geometry_fail_safe(self):
        for key in ('image','source_sha256','shape'):
            old,a,b=baseline(),case('a'),case('b')
            if key=='shape':b['predictions']['source_shape']=[1,1]
            else:b[key]='other'
            result=append_strong_consensus(old,a,b)
            self.assertEqual(result['all_predictions'],old['all_predictions'])
            self.assertIsNotNone(result['strong_consensus_fallback_reason'])
    def test_single_view_or_repeated_tile_not_confirmation(self):
        for rows in ([row(1500)], [row(1500),row(1500)]):
            a,b=case('a'),case('b');b['predictions']['edge_kept_predictions']=rows
            self.assertEqual(append_strong_consensus(baseline(),a,b)['strong_consensus_additions'],[])
    def test_threshold_strict_and_nonfinite_disallowed(self):
        for score in (.9, float('nan'),float('inf')):
            a,b=case('a'),case('b');b['predictions']['merged_predictions'][0]['confidence']=score
            self.assertEqual(append_strong_consensus(baseline(),a,b)['strong_consensus_additions'],[])
    def test_class_agnostic_but_cross_class_agreement_rejected(self):
        for cls in (0,1):
            self.assertEqual(len(append_strong_consensus(baseline(),case('a',cls=cls),case('b',cls=cls))['strong_consensus_additions']),1)
        self.assertEqual(append_strong_consensus(baseline(),case('a'),case('b',cls=1))['strong_consensus_additions'],[])
    def test_position_disagreement_rejected(self):
        self.assertEqual(append_strong_consensus(baseline(),case('a'),case('b',x=1530))['strong_consensus_additions'],[])
    def test_shared_budget_and_duplicate_keep_old(self):
        old=baseline();old['all_predictions'] += [row(700+i*60) for i in range(5)]
        self.assertEqual(append_strong_consensus(old,case('a'),case('b'))['strong_consensus_additions'],[])
        self.assertEqual(append_strong_consensus(baseline(),case('a',x=100),case('b',x=100))['strong_consensus_additions'],[])
    def test_bad_old_budget_fails_closed(self):
        old=baseline();old['all_predictions'] += [row(700+i*60) for i in range(6)]
        self.assertIsNotNone(append_strong_consensus(old,case('a'),case('b'))['strong_consensus_fallback_reason'])
    def test_both_crops_can_confirm_but_same_crop_cannot(self):
        def rechecked(weight, same):
            c=case(weight);c['predictions']['merged_predictions']=[];c['predictions']['edge_kept_predictions']=[]
            a,b=row(1500),row(1500)
            windows=[[0,0,1280,1280],[0,0,1280,1280] if same else [200,200,1480,1480]]
            c['zoom_evidence']=[dict(proposal=row(1500),views=[[a],[b]],windows=windows)]
            return c
        self.assertEqual(len(append_strong_consensus(baseline(),rechecked('a',False),rechecked('b',False))['strong_consensus_additions']),1)
        self.assertEqual(append_strong_consensus(baseline(),rechecked('a',False),rechecked('b',True))['strong_consensus_additions'],[])


if __name__=='__main__':unittest.main()
