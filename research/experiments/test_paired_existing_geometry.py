import copy
import unittest
from paired_existing_geometry import proposals,refine


class ExistingGeometry(unittest.TestCase):
    def model(self,weight,box):
        return dict(source_sha256='s',weight_sha256=weight,predictions=dict(source_shape=[300,300],merged_predictions=[dict(class_id=0,confidence=.9,box_xyxy=box)]))
    def test_three_unique_weights_not_views(self):
        old = dict(class_id=0,confidence=.7,box_xyxy=[100,100,150,140]); current = dict(primary=[old],all_predictions=[old])
        models = [self.model(w,[102,100,152,140]) for w in ('a','b','b')]
        self.assertEqual(proposals(models[0],models,current),[])
        models[-1]['weight_sha256'] = 'c'
        candidates = proposals(models[0],models,current); self.assertEqual(len(candidates),1)
        before = copy.deepcopy(current); trial = refine(current,candidates,[[.001,.998,.001]],'h')
        self.assertEqual(len(trial['all_predictions']),1)
        self.assertEqual(trial['all_predictions'][0]['confidence'],.7)
        self.assertEqual(current,before)
        self.assertEqual(trial['primary'],before['primary'])
    def test_low_probability_keeps_original(self):
        old = dict(class_id=0,confidence=.7,box_xyxy=[100,100,150,140]); current = dict(primary=[old],all_predictions=[old])
        models = [self.model(w,[102,100,152,140]) for w in ('a','b','c')]
        self.assertEqual(refine(current,proposals(models[0],models,current),[[.2,.799,.001]],'h')['all_predictions'],[old])


if __name__ == '__main__': unittest.main()
