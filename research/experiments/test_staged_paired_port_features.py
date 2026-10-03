import importlib.util
import sys
import unittest
from pathlib import Path
sys.path.insert(0,'E:/PythonProject10')
import numpy as np
import torch
import port_semantic_verifier as original_embeddings
import paired_port_semantics as original_geometry
from paired_port_semantic_selection import select as original_select
from port_semantic_model_vote import proposals as original_proposals
spec=importlib.util.spec_from_file_location('staged_paired_features',Path(__file__).resolve().parents[1]/'staging/paired_geometry_release/paired_port_features.py')
staged=importlib.util.module_from_spec(spec);spec.loader.exec_module(staged)


class StagedParityTests(unittest.TestCase):
    def test_crop_preprocessing_reference_warp_and_valid_context_exact(self):
        rng=np.random.default_rng(4);image=rng.integers(0,256,(200,300,3),dtype=np.uint8)
        boxes=[[1,1,20,30],[100,80,145,100],[270,90,295,150]]
        for box in boxes:
            for scale in (1.5,3.):self.assertTrue(torch.equal(staged.crop_tensor(image,box,scale),original_embeddings.crop_tensor(image,box,scale)))
        matrix=np.array([[1,0,10],[0,1,-3],[0,0,1]],float)
        a=staged.expected_in_source(image,matrix,image.shape[:2]);b=original_geometry.expected_in_source(image,matrix,image.shape[:2])
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(a,b)))
        self.assertEqual(staged.valid_boxes(boxes,a[1]),original_geometry.valid_boxes(boxes,b[1]))

    def test_descriptor_pooling_and_pair_channels_exact(self):
        class Encoder:
            def forward_features(self,tensor):
                values=tensor.mean(dim=(1,2,3)).view(-1,1)
                cls=values*torch.arange(1,385).view(1,384)
                patches=(cls[:,None,:]*torch.arange(1,257).view(1,256,1))
                return dict(x_norm_clstoken=cls,x_norm_patchtokens=patches)
        image=np.full((200,300,3),120,np.uint8);boxes=[[100,80,145,100]]
        a=staged.embeddings(Encoder(),image,boxes);b=original_embeddings.embeddings(Encoder(),image,boxes)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(staged.paired_features(a,b),original_geometry.paired_features(a,b)))
        self.assertEqual(tuple(staged.embeddings(Encoder(),image,[]).shape),(0,1536))

    def test_native_proposal_and_selector_exact(self):
        row=dict(class_id=0,box_xyxy=[100,100,150,150],confidence=.1)
        models=[dict(source_sha256='same',weight_sha256=digest,predictions=dict(source_shape=[300,400],merged_predictions=[row])) for digest in ('t','s')]
        a=staged.proposals(models[0],models);b=original_proposals(models[0],models)
        self.assertEqual(a,b);current=dict(primary=[],all_predictions=[])
        self.assertEqual(staged.select(current,a,[[.01,.98,.01]],'head'),original_select(current,b,[[.01,.98,.01]],'head'))


if __name__=='__main__':unittest.main()
