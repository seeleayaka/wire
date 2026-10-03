import copy
import sys
import unittest
sys.path.insert(0,'E:/PythonProject10')
import numpy as np
from inspection_agent.optional_port_crop_review import aligned_predictions
from paired_graph_hint_link import accepted_native_rows


class HintLinkTests(unittest.TestCase):
    def test_native_geometry_preserved_despite_nonidentity_warp(self):
        rows=[dict(class_id=0,confidence=.91,box_xyxy=[100.25,120.5,150.75,170.5])];before=copy.deepcopy(rows)
        matrix=np.array([[1.1,0,7],[0,1.05,8],[0,0,1]],float)
        mapped=aligned_predictions([dict(rows[0],support_tiles=[])],matrix,[480,640],[480,640])
        result=accepted_native_rows(rows,[dict(box=mapped[0])],matrix,[480,640],[480,640])
        self.assertEqual(result,before);self.assertEqual(rows,before)
    def test_bad_mapping_and_duplicate_hint_fail_closed(self):
        rows=[dict(class_id=0,confidence=.91,box_xyxy=[100.,120.,150.,170.])];matrix=np.eye(3)
        mapped=aligned_predictions([dict(rows[0],support_tiles=[])],matrix,[480,640],[480,640])
        hint=dict(box=mapped[0]);bad=copy.deepcopy(hint);bad['box']['right']+=1
        with self.assertRaises(ValueError):accepted_native_rows(rows,[bad],matrix,[480,640],[480,640])
        with self.assertRaises(ValueError):accepted_native_rows(rows,[hint,hint],matrix,[480,640],[480,640])


if __name__=='__main__':unittest.main()
