"""Exercise actual installed processor method with synthetic model tensors.

Extracting the method avoids a second SAM model/import graph while inference runs.
The model is a fixture, not a real-photo accuracy or decoder reproducibility test.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
import torch.nn.functional as F

PROCESSOR = Path('E:/PythonProject10/runtime/sam3/source/sam3/model/sam3_image_processor.py')


def installed_forward():
    tree = ast.parse(PROCESSOR.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Sam3Processor')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_forward_grounding')
    module = ast.Module(body=[node], type_ignores=[])
    namespace = dict(torch=torch, Dict=dict, interpolate=F.interpolate,
        box_ops=SimpleNamespace(box_cxcywh_to_xyxy=lambda b: torch.cat((b[...,:2]-b[...,2:]/2,
                                                                       b[...,:2]+b[...,2:]/2),dim=-1)))
    exec(compile(module,str(PROCESSOR),'exec'),namespace)
    return namespace['_forward_grounding']


class NativeThresholdTests(unittest.TestCase):
    def forward(self, threshold):
        outputs = dict(pred_boxes=torch.tensor([[[.5,.5,.5,.5],[.25,.25,.2,.2]]]),
            pred_logits=torch.logit(torch.tensor([[[.9],[.4]]])),
            pred_masks=torch.tensor([[[[2.,-2.],[-2.,2.]],[[1.,1.],[-1.,-1.]]]]),
            presence_logit_dec=torch.tensor([20.]))
        model = SimpleNamespace(forward_grounding=lambda **kw:outputs)
        processor = SimpleNamespace(model=model,find_stage=None,confidence_threshold=threshold,device='cpu')
        state = dict(backbone_out={},geometric_prompt=None,original_height=2,original_width=2)
        return installed_forward()(processor,state)

    def test_lower_collection_does_not_repair_high_score_pixels(self):
        low,high = self.forward(.3),self.forward(.5)
        self.assertEqual(len(low['scores']),2)
        self.assertEqual(len(high['scores']),1)
        self.assertTrue(torch.equal(low['masks'][0],high['masks'][0]))
        self.assertTrue(torch.equal(low['masks_logits'][0],high['masks_logits'][0]))
        self.assertTrue(torch.equal(low['boxes'][0],high['boxes'][0]))

    def test_threshold_is_strict_and_pixel_cut_is_separate(self):
        result = self.forward(.9)
        self.assertEqual(len(result['scores']),0)
        native = self.forward(.3)
        self.assertTrue(torch.equal(native['masks'],native['masks_logits']>.5))


if __name__ == '__main__': unittest.main()
