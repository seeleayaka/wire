import copy
import sys
import unittest
sys.path.insert(0,'E:/PythonProject10')
from paired_semantic_components import unique_components


def row(box,confidence=.8):return dict(class_id=0,box_xyxy=box,confidence=confidence)

def model(digest,rows):return dict(source_sha256='same',weight_sha256=digest,predictions=dict(source_shape=[600,800],merged_predictions=rows))

def candidate(native,digest,score):
    return dict(native,proposal_detector_score=native['confidence'],confidence=score,semantic_detector_weight_sha256=digest)


class ComponentsTests(unittest.TestCase):
    def test_two_boxes_share_one_opposite_witness_without_NMS_relaxation(self):
        a=row([100,100,200,160]);b=row([130,100,230,160]);w=row([115,100,215,160])
        teacher=model('teacher',[a,b]);student=model('student',[w])
        additions=[candidate(a,'teacher',.999),candidate(b,'teacher',.99)];before=copy.deepcopy(additions)
        kept,audit=unique_components(additions,teacher,student)
        self.assertEqual(kept,[additions[0]]);self.assertTrue(audit[1]['suppressed_shared_detection_component'])
        self.assertEqual(additions,before)

    def test_distinct_components_retained_and_unlinked_fails_closed(self):
        a=row([100,100,200,160]);b=row([300,100,400,160])
        teacher=model('teacher',[a,b]);student=model('student',[copy.deepcopy(a),copy.deepcopy(b)])
        additions=[candidate(a,'teacher',.999),candidate(b,'teacher',.99)]
        self.assertEqual(len(unique_components(additions,teacher,student)[0]),2)
        additions[0]['box_xyxy']=[110,100,200,160]
        with self.assertRaises(ValueError):unique_components(additions,teacher,student)


if __name__=='__main__':unittest.main()
