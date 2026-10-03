import unittest
from paired_pose_native_training import curate,native_label,pixel_signature


class NativeTrainingTest(unittest.TestCase):
    def row(self,**kw):
        return dict(dict(image='synthetic',fold=0,box=[100,100,120,120],label=0,kind='native'),**kw)

    def test_duplicate_and_conflict_drop(self):
        rows=[self.row(),self.row(),self.row(label=1),self.row(image='another',fold=1)]
        kept,dup,conflict=curate(rows,{'synthetic':0,'another':1})
        self.assertEqual(kept,[3]); self.assertEqual(dup,[]); self.assertEqual(conflict,[0,1,2])

    def test_reference_and_source_are_distinct(self):
        a,b=self.row(),self.row(kind='reference_self')
        self.assertNotEqual(pixel_signature(a),pixel_signature(b))
        self.assertEqual(curate([a,a,b],{'synthetic':0})[:2],([0,2],[(0,1)]))

    def test_wrong_source_fold_rejected(self):
        with self.assertRaises(ValueError):curate([self.row(fold=1)],{'synthetic':0})

    def test_training_label_class_and_iou(self):
        proposal=dict(class_id=0,box_xyxy=[0,0,10,10])
        self.assertEqual(native_label(proposal,[dict(class_id=0,box=[0,0,10,10])]),1)
        self.assertEqual(native_label(proposal,[dict(class_id=1,box=[0,0,10,10])]),0)
        self.assertEqual(native_label(proposal,[dict(class_id=0,box=[10,10,20,20])]),0)


if __name__=='__main__':unittest.main()
