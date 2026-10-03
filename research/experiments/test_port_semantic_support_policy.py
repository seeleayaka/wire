import copy,unittest
from test_teacher_student_port_policy import fixtures,case
from teacher_student_port_policy import merge
from port_semantic_support_policy import proposals,append_semantic
class SemanticPolicyTests(unittest.TestCase):
    def fixture(self):
        teacher,new=fixtures();current=merge(teacher,case([]))
        return teacher,new,current
    def test_preserve_prefix_and_inputs(self):
        teacher,new,current=self.fixture();before=copy.deepcopy((teacher,new,current))
        candidates=proposals(teacher,[new]);self.assertEqual(len(candidates),1)
        output=append_semantic(current,candidates,[[.001,.998,.001]])
        self.assertEqual(len(output['semantic_additions']),1)
        self.assertEqual(output['all_predictions'][:len(current['all_predictions'])],current['all_predictions'])
        self.assertEqual((teacher,new,current),before)
    def test_one_tile_fails(self):
        teacher,new,current=self.fixture();new['predictions']['edge_kept_predictions'][1]['source_tile']=0
        self.assertEqual(proposals(teacher,[new]),[])
    def test_source_mismatch_rejected(self):
        teacher,new,current=self.fixture();new['source_sha256']='changed'
        with self.assertRaises(ValueError):proposals(teacher,[new])
    def test_teacher_support_still_required(self):
        teacher,new,current=self.fixture()
        teacher['predictions']['merged_predictions']=[r for r in teacher['predictions']['merged_predictions'] if r['confidence']>.5]
        self.assertEqual(proposals(teacher,[new]),[])
    def test_wrong_class_or_low_probability_fails(self):
        teacher,new,current=self.fixture();candidates=proposals(teacher,[new])
        for p in ([.001,.979,.020],[.001,.001,.998]):
            self.assertEqual(append_semantic(current,candidates,[p])['semantic_additions'],[])
    def test_invalid_probabilities_rejected(self):
        teacher,new,current=self.fixture();candidates=proposals(teacher,[new])
        for p in ([0.,float('nan'),0.],[-.5,1.,.5],[.001,.998,.998],[.99]):
            with self.assertRaises(ValueError):append_semantic(current,candidates,[p])
    def test_full_budget(self):
        teacher,new=fixtures(True);current=merge(teacher,case([]))
        self.assertEqual(append_semantic(current,proposals(teacher,[new]),[[.001,.998,.001]])['semantic_additions'],[])
    def test_duplicate_cues(self):
        teacher,new,current=self.fixture();current=merge(teacher,new)
        self.assertEqual(append_semantic(current,proposals(teacher,[new]),[[.001,.998,.001]])['semantic_additions'],[])
if __name__=='__main__':unittest.main()
