import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from core import image_binding
from run_prompt_contrast import write_masks, digest
from run_review import verified_run, read
from analyze_prompt_contrast import overlaps


class PromptContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source.png'
        self.image = Image.new('RGB',(24,20),'white')
        self.image.save(self.source)
        self.binding = dict(image_binding(self.source),path=str(self.source))

    def write(self, masks, scores=None, directory='run'):
        scores=np.full(len(masks),.8) if scores is None else np.asarray(scores)
        run=self.root/directory
        write_masks(run,self.image,masks,np.zeros((len(masks),4)),scores,'wire',1,1,
                    self.binding,{},1)
        return run

    def test_empty_inventory_is_bound(self):
        run=self.write(np.zeros((0,20,24),bool))
        origin=verified_run(run,self.source)
        self.assertEqual(origin['paths'],[])
        self.assertFalse(read(run/'sam/report.json')['image_state_cache_reused'])

    def test_inventory_beyond_99_pinned(self):
        run=self.write(np.zeros((104,20,24),bool))
        origin=verified_run(run,self.source)
        self.assertEqual(len(origin['paths']),104)
        self.assertEqual(read(run/'sam/report.json')['model_observer_count'],1)

    def test_output_never_overwritten(self):
        masks=np.zeros((1,20,24),bool)
        run=self.write(masks)
        prior=digest(run/'run_manifest.json')
        with self.assertRaises(FileExistsError):
            self.write(masks)
        self.assertEqual(prior,digest(run/'run_manifest.json'))

    def test_bad_shape_rejected(self):
        with self.assertRaisesRegex(ValueError,'shape'):
            self.write(np.zeros((1,21,24),bool))

    def test_score_gate_and_nonfinite_rejected(self):
        for i,score in enumerate([.5,1.1,float('nan')]):
            with self.subTest(score=score),self.assertRaises(ValueError):
                self.write(np.zeros((1,20,24),bool),[score],str(i))

    def test_snapshot_drift_rejected(self):
        self.image.resize((25,20)).save(self.source)
        with self.assertRaisesRegex(ValueError,'drift'):
            self.write(np.zeros((1,20,24),bool))

    def test_overlap_is_not_one_to_one_or_truth(self):
        a=np.zeros((20,24),bool);a[10,2:12]=True
        b=a.copy();b[9,2:12]=True
        result=overlaps([a,a],[b])
        self.assertEqual(result,[{'best_target_id':1,'best_IoU':.5}]*2)
        self.assertEqual(overlaps([a],[]),[{'best_target_id':None,'best_IoU':0.}])


if __name__=='__main__':
    unittest.main()
