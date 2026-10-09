from copy import deepcopy
import unittest
from exact_paired_score_cache import binding,reuse


class ExactCacheTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(class_id=0,box_xyxy=[10.,20.,30.,40.],semantic_model_vote_sha256=['old'])
        self.prob=[.01,.985,.005]
        self.runtime=dict(encoder='e'*64,head='f'*64,accepted={'crop_recipe':'v1'})
        self.alignment=dict(alignment_quality={'reliable':True},source_to_reference_homography=[[1,0,0],[0,1,0],[0,0,1]])
        self.args=['a'*64,'b'*64,self.runtime,self.alignment,[2736,3648]]
        self.digest=binding(*self.args)

    def test_exact_same_input_reused_not_an_extra_vote(self):
        row=deepcopy(self.row);row['semantic_model_vote_sha256']=['new']
        scores,missing,hits=reuse([row],[self.row],[self.prob],self.digest,self.digest)
        self.assertEqual(scores,[self.prob]);self.assertEqual(missing,[]);self.assertEqual(hits,1)
        self.assertEqual(row['semantic_model_vote_sha256'],['new'])

    def test_nearby_box_never_reused(self):
        row=deepcopy(self.row);row['box_xyxy'][0]+=1e-8
        self.assertEqual(reuse([row],[self.row],[self.prob],self.digest,self.digest)[1],[0])

    def test_class_change_never_reused(self):
        row=deepcopy(self.row);row['class_id']=1
        self.assertEqual(reuse([row],[self.row],[self.prob],self.digest,self.digest)[1],[0])

    def test_pixel_warp_runtime_or_shape_change_invalidates(self):
        for index in range(5):
            args=deepcopy(self.args)
            if index<2:args[index]='c'*64
            elif index==2:args[index]['accepted']['crop_recipe']='v2'
            elif index==3:args[index]['source_to_reference_homography'][0][2]=1e-10
            else:args[index][0]+=1
            with self.subTest(index=index):
                other=binding(*args);self.assertNotEqual(other,self.digest)
                self.assertEqual(reuse([self.row],[self.row],[self.prob],self.digest,other)[1],[0])

    def test_inference_order_and_new_positions_preserved(self):
        other=deepcopy(self.row);other['box_xyxy']=[50,60,70,80]
        scores,missing,hits=reuse([other,self.row],[self.row],[self.prob],self.digest,self.digest)
        self.assertEqual(scores,[None,self.prob]);self.assertEqual(missing,[0]);self.assertEqual(hits,1)

    def test_duplicate_contradiction_or_invalid_probabilities_rejected(self):
        with self.assertRaises(ValueError):reuse([self.row],[self.row,self.row],[self.prob,[.02,.975,.005]],self.digest,self.digest)
        for probabilities in [[],[[.1,.9]],[[float('nan'),0,1]],[[True,0,0]],[[.1,.9,.1]]]:
            with self.subTest(probabilities=probabilities),self.assertRaises(ValueError):
                reuse([self.row],[self.row],probabilities,self.digest,self.digest)

    def test_inputs_and_probability_outputs_not_aliased(self):
        before=deepcopy(self.row);prob=list(self.prob)
        result=reuse([self.row],[self.row],[prob],self.digest,self.digest)[0]
        result[0][1]=.2
        self.assertEqual(prob,self.prob);self.assertEqual(self.row,before)

    def test_unreliable_or_nonfinite_binding_rejects(self):
        for bad in [False,None]:
            args=deepcopy(self.args);args[3]['alignment_quality']['reliable']=bad
            with self.assertRaises(ValueError):binding(*args)
        args=deepcopy(self.args);args[3]['source_to_reference_homography'][0][0]=float('nan')
        with self.assertRaises(ValueError):binding(*args)


if __name__=='__main__':unittest.main(verbosity=2)
