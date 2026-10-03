import copy
import sys
import unittest
sys.path.insert(0,'E:/PythonProject10')
import numpy as np
from paired_semantic_reference_veto import veto


class ReferenceVetoTests(unittest.TestCase):
    def test_exact_original_reference_threshold_class_and_geometry(self):
        candidates=[dict(box_xyxy=[100,100,150,150],class_id=0,confidence=.99)]
        ref=dict(left=100,top=100,right=150,bottom=150,class_id=0,confidence=.25)
        before=copy.deepcopy(candidates)
        self.assertEqual(len(veto(candidates,[ref],np.eye(3),[300,400],[300,400])[0]),1)
        ref['confidence']=.25001
        self.assertEqual(veto(candidates,[ref],np.eye(3),[300,400],[300,400])[0],[])
        ref['class_id']=1
        self.assertEqual(veto(candidates,[ref],np.eye(3),[300,400],[300,400])[0],before)
        self.assertEqual(candidates,before)

    def test_reference_requires_correct_warp_and_never_backfills(self):
        candidate=dict(box_xyxy=[100,100,150,150],class_id=0,confidence=.99)
        ref=dict(left=150,top=100,right=200,bottom=150,class_id=0,confidence=.9)
        matrix=np.array([[1,0,50],[0,1,0],[0,0,1]],float)
        kept,audit=veto([candidate],[ref],matrix,[300,400],[300,400])
        self.assertEqual(kept,[]);self.assertTrue(audit[0]['rejected'])
        self.assertEqual(veto([],[],matrix,[300,400],[300,400]),([],[]))


if __name__=='__main__':unittest.main()
