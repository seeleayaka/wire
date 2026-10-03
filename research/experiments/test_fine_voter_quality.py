import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path('E:/PythonProject10')))
import unittest
from fine_voter_quality import attach


class FineVoterTests(unittest.TestCase):
    def model(self,weight,box,cls=0,score=.8):
        return dict(weight_sha256=weight,source_sha256='source',predictions=dict(source_shape=[1000,1000],merged_predictions=[dict(box_xyxy=box,class_id=cls,confidence=score)]))

    def candidate(self):return dict(box_xyxy=[100,100,200,200],class_id=0,semantic_model_vote_sha256=['a','b','c'])

    def test_repeated_views_one_checkpoint_vote(self):
        models=[self.model(w,[100,100,200,200]) for w in ('a','a','b','c')]
        result=attach([self.candidate()],models,'source',(1000,1000))
        self.assertEqual(result[0]['localization_voter_best_IoU'],{'a':1.,'b':1.,'c':1.})

    def test_best_geometry_from_actual_view(self):
        models=[self.model('a',[90,90,210,210]),self.model('a',[100,100,200,200]),self.model('b',[100,100,200,200]),self.model('c',[100,100,200,200])]
        self.assertEqual(attach([self.candidate()],models,'source',(1000,1000))[0]['localization_voter_best_IoU']['a'],1.)

    def test_wrong_source_and_shape_rejected(self):
        models=[self.model(w,[100,100,200,200]) for w in ('a','b','c')]
        with self.assertRaises(ValueError):attach([self.candidate()],models,'different',(1000,1000))
        with self.assertRaises(ValueError):attach([self.candidate()],models,'source',(900,1000))

    def test_class_floor_and_missing_votes_rejected(self):
        for bad in (self.model('c',[100,100,200,200],1),self.model('c',[100,100,200,200],score=.05),self.model('c',[5,5,10,10])):
            models=[self.model(w,[100,100,200,200]) for w in ('a','b')]+[bad]
            with self.assertRaises(ValueError):attach([self.candidate()],models,'source',(1000,1000))

    def test_inputs_immutable(self):
        models=[self.model(w,[100,100,200,200]) for w in ('a','b','c')];c=[self.candidate()]
        oldmodels=copy.deepcopy(models);oldc=copy.deepcopy(c);attach(c,models,'source',(1000,1000))
        self.assertEqual(models,oldmodels);self.assertEqual(c,oldc)

    def test_scale_translation_IoU_invariance(self):
        models=[self.model(w,[100,100,200,200]) for w in ('a','b','c')];c=[self.candidate()]
        before=attach(c,models,'source',(1000,1000))
        for row in [*c,*[m['predictions']['merged_predictions'][0] for m in models]]:
            a=row['box_xyxy'];row['box_xyxy']=[a[0]*2+30,a[1]*.5+40,a[2]*2+30,a[3]*.5+40]
        after=attach(c,models,'source',(1000,1000))
        self.assertEqual(before[0]['localization_voter_best_IoU'],after[0]['localization_voter_best_IoU'])


if __name__=='__main__':unittest.main()
