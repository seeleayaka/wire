"""General geometry and safety invariants, independent of dataset photo IDs."""
import copy
import math
import random
import unittest
from inspection_agent.paired_native_pose_features import proposals,poses,select


class NativePoseContract(unittest.TestCase):
    def model(self,weight,box=(100,100,150,140),cls=0,shape=(400,500)):
        return dict(source_sha256='synthetic_source',weight_sha256=weight,predictions=dict(source_shape=list(shape),
            merged_predictions=[dict(class_id=cls,confidence=.8,box_xyxy=list(box))]))

    def empty(self):return dict(primary=[],all_predictions=[])

    def candidate(self,parent=0,box=(100,100,150,140)):
        return dict(class_id=0,confidence=.8,box_xyxy=list(box),semantic_model_vote_sha256=['a','b','c'],pose_parent_seed_id=parent)

    def test_seven_uniform_poses_leave_input_unchanged(self):
        box=[100,100,150,140];before=copy.deepcopy(box)
        self.assertEqual(len(poses(box)),7);self.assertEqual(box,before)

    def test_nonpositive_seed_rejected(self):
        for box in ([100,100,100,140],[150,100,100,140],[100,140,150,100]):
            with self.subTest(box=box),self.assertRaises(ValueError):poses(box)

    def test_same_checkpoint_views_cannot_make_three_votes(self):
        row=self.candidate();row['semantic_model_vote_sha256']=['a','b','a','b']
        self.assertEqual(select(self.empty(),[row],[[.001,.998,.001]],'h')['all_predictions'],[])

    def test_same_source_shape_required(self):
        models=[self.model(w) for w in ('a','b','c')];models[-1]['predictions']['source_shape']=[401,500]
        with self.assertRaises(ValueError):proposals(models[0],models,self.empty())

    def test_same_source_fingerprint_required(self):
        models=[self.model(w) for w in ('a','b','c')];models[-1]['source_sha256']='other_source'
        with self.assertRaises(ValueError):proposals(models[0],models,self.empty())

    def test_different_classes_do_not_vote_for_each_other(self):
        models=[self.model('a'),self.model('b',cls=1),self.model('c',cls=1)]
        rows=proposals(models[0],models,self.empty())
        self.assertEqual(select(self.empty(),rows,[[.001,.998,.001]]*len(rows),'h')['all_predictions'],[])

    def test_same_box_already_reviewed_is_not_added(self):
        models=[self.model(w) for w in ('a','b','c')];old=self.empty();old['all_predictions']=[models[0]['predictions']['merged_predictions'][0]]
        self.assertEqual(proposals(models[0],models,old),[])

    def test_probability_count_must_match_proposals(self):
        with self.assertRaises(ValueError):select(self.empty(),[self.candidate()],[],'h')

    def test_invalid_probabilities_rejected(self):
        for score in ([math.nan,.999,.001],[math.inf,.999,.001],[-.01,1.01,0],[.01,.98],[.1,.98,.1]):
            with self.subTest(score=score),self.assertRaises(ValueError):select(self.empty(),[self.candidate()],[score],'h')

    def test_exact_fixed_p98_gate(self):
        row=self.candidate()
        self.assertEqual(len(select(self.empty(),[row],[[.01,.98,.01]],'h')['all_predictions']),1)
        self.assertEqual(select(self.empty(),[row],[[.010001,.979999,.01]],'h')['all_predictions'],[])

    def test_wrong_classifier_class_is_not_added(self):
        self.assertEqual(select(self.empty(),[self.candidate()],[[.001,.001,.998]],'h')['all_predictions'],[])

    def test_one_pose_per_parent_uses_highest_fixed_score(self):
        low=self.candidate();high=self.candidate(box=(110,100,160,140))
        out=select(self.empty(),[low,high],[[.005,.99,.005],[.001,.998,.001]],'h')
        self.assertEqual(len(out['all_predictions']),1);self.assertEqual(out['all_predictions'][0]['box_xyxy'],high['box_xyxy'])

    def test_candidate_order_does_not_change_result(self):
        rows=[self.candidate(parent=i,box=(50+i*80,100,90+i*80,140)) for i in range(4)]
        before=select(self.empty(),rows,[[.001,.998,.001]]*4,'h')
        random.Random(20261004).shuffle(rows)
        self.assertEqual(before,select(self.empty(),rows,[[.001,.998,.001]]*4,'h'))

    def test_full_shared_extra_budget_adds_nothing(self):
        rows=[self.candidate(parent=i,box=(50+i*80,100,90+i*80,140)) for i in range(5)]
        old=dict(primary=[],all_predictions=copy.deepcopy(rows));before=copy.deepcopy(old)
        out=select(old,[self.candidate(parent=9,box=(50,200,90,240))],[[.001,.998,.001]],'h')
        self.assertEqual(out['all_predictions'],before['all_predictions']);self.assertEqual(old,before)

    def test_one_remaining_slot_does_not_expand_budget(self):
        old=dict(primary=[],all_predictions=[self.candidate(parent=i,box=(50+i*80,100,90+i*80,140)) for i in range(4)])
        extra=[self.candidate(parent=8,box=(50,200,90,240)),self.candidate(parent=9,box=(150,200,190,240))]
        out=select(old,extra,[[.005,.99,.005],[.001,.998,.001]],'h')
        self.assertEqual(len(out['all_predictions']),5);self.assertEqual(out['all_predictions'][:4],old['all_predictions'])

    def test_translation_covariance_away_from_image_border(self):
        models=[self.model(w) for w in ('a','b','c')];baseline=proposals(models[0],models,self.empty())
        moved=[self.model(w,box=(130,150,180,190)) for w in ('a','b','c')]
        result=proposals(moved[0],moved,self.empty())
        self.assertEqual(len(baseline),len(result))
        for old,new in zip(baseline,result):
            self.assertEqual(new['box_xyxy'],[old['box_xyxy'][0]+30,old['box_xyxy'][1]+50,old['box_xyxy'][2]+30,old['box_xyxy'][3]+50])

    def test_uniform_scale_covariance_away_from_image_border(self):
        models=[self.model(w) for w in ('a','b','c')];baseline=proposals(models[0],models,self.empty())
        scaled=[self.model(w,box=(200,200,300,280),shape=(800,1000)) for w in ('a','b','c')]
        result=proposals(scaled[0],scaled,self.empty());self.assertEqual(len(result),len(baseline))
        for old,new in zip(baseline,result):self.assertEqual(new['box_xyxy'],[v*2 for v in old['box_xyxy']])

    def test_proposal_construction_does_not_mutate_detector_or_current(self):
        models=[self.model(w) for w in ('a','b','c')];old=self.empty();before=copy.deepcopy((models,old))
        rows=proposals(models[0],models,old);select(old,rows,[[.001,.998,.001]]*len(rows),'h')
        self.assertEqual((models,old),before)


if __name__=='__main__':unittest.main()
