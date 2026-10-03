import copy
import unittest
from tools.evaluate_anchored_local import anchored_selection

class AnchorTests(unittest.TestCase):
    def box(self,x,rank):
        return {'left':x,'top':0,'right':x+1,'bottom':1,'rank':rank}

    def select(self,baseline,pool):
        return anchored_selection(baseline,pool,None,10,10,lambda b,*_:b['rank'])

    def test_primary_evidence_survives_higher_novelty_scores(self):
        anchor=self.box(0,0); old=self.box(1,1); novel=self.box(2,100)
        self.assertEqual(self.select([anchor,old],[anchor,old,novel]),[anchor,novel])

    def test_single_slot_and_empty_baseline_preserve_original_budget(self):
        anchor=self.box(0,0); novel=self.box(2,100)
        self.assertEqual(self.select([anchor],[anchor,novel]),[anchor])
        self.assertEqual(self.select([],[novel]),[])

    def test_duplicate_geometry_does_not_consume_slots_or_mutate_input(self):
        anchor=self.box(0,0); duplicate=self.box(0,100); novel=self.box(2,90)
        pool=[anchor,duplicate,novel]
        before=copy.deepcopy(pool)
        self.assertEqual(self.select([anchor,novel],pool),[anchor,novel])
        self.assertEqual(pool,before)

if __name__=='__main__':
    unittest.main()
