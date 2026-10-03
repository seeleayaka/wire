import unittest
import torch
from paired_pose_native_final_backend import scored_features,validate_checkpoint


class NativeFinalScoring(unittest.TestCase):
    def test_actual_median_fingerprint_contract(self):
        checkpoint=dict(input_dimensions=6144,classes=['other','unplugged_plug','unplugged_jack'],encoder_sha256='bound')
        validate_checkpoint(checkpoint,dict(accepted={},encoder='bound'))
        with self.assertRaises(ValueError):validate_checkpoint(checkpoint,dict(accepted={},encoder='changed'))

    def test_uses_new_head_without_old_score_boost_or_veto(self):
        head=torch.nn.Linear(6144,3)
        with torch.no_grad():head.weight.zero_();head.bias.copy_(torch.tensor([-8.,8.,-8.]))
        row=dict(class_id=0,confidence=.1,box_xyxy=[100,100,150,140],semantic_model_vote_sha256=['a','b','c'],pose_parent_seed_id=0)
        current=dict(primary=[],all_predictions=[])
        result,scores=scored_features(current,[row],[[1,0,0]],torch.zeros((1,6144)),head)
        self.assertEqual(len(result['all_predictions']),1)
        self.assertEqual(result['all_predictions'][0]['confidence'],scores[0][1])

    def test_exact_native_feature_count_required(self):
        with self.assertRaises(ValueError):scored_features({},[{}],[],torch.zeros((0,6144)),torch.nn.Linear(6144,3))


if __name__=='__main__':unittest.main()
