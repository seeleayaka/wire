import unittest
from copy import deepcopy
from protected_allport_baseline import check_accepted,check_totals


class ProtectedBaselineTests(unittest.TestCase):
    def setUp(self):
        cue=lambda i:dict(class_id=0,box_xyxy=[i,1,i+1,2],confidence=.99)
        selection=lambda rows:dict(primary=rows[:1],all_predictions=rows)
        self.legacy=selection([cue(1)])
        self.accepted=selection([cue(1),cue(2)])
        self.research=selection([cue(1),cue(2),cue(3)])

    def test_exact_three_generation_prefix(self):
        self.assertTrue(check_accepted(self.legacy,self.accepted,self.research,self.research))

    def test_accepted_row_loss_cannot_hide_behind_legacy_prefix(self):
        trial=deepcopy(self.research);trial['all_predictions'][1]['box_xyxy'][0]+=.01
        with self.assertRaises(ValueError):check_accepted(self.legacy,self.accepted,self.research,trial)

    def test_explicit_legacy277_vs_accepted295(self):
        metric=lambda tp:dict(tp=tp,unmatched=4,fn=344-tp,predictions=tp+4,targets=344)
        totals={k:metric(v) for k,v in [('original',277),('accepted',295),('current',298)]}
        self.assertTrue(check_totals(totals,legacy=True))
        with self.assertRaises(ValueError):check_totals(totals)

    def test_wrong_accepted_snapshot_rejected(self):
        metric=lambda tp:dict(tp=tp,unmatched=4,fn=344-tp,predictions=tp+4,targets=344)
        with self.assertRaises(ValueError):check_totals(dict(original=metric(277),accepted=metric(277),current=metric(298)),legacy=True)

    def test_correct_child_original295(self):
        metric=lambda tp:dict(tp=tp,unmatched=4,fn=344-tp,predictions=tp+4,targets=344)
        self.assertTrue(check_totals(dict(original=metric(295),current=metric(298))))

    def test_count_only_identity_does_not_pass(self):
        trial=deepcopy(self.research);trial['all_predictions'].reverse()
        with self.assertRaises(ValueError):check_accepted(self.legacy,self.accepted,self.research,trial)
