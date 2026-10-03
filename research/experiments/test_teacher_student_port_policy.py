import copy,unittest
from teacher_student_port_policy import merge

def row(x,score=.9,cls=0,tile=0):return dict(box_xyxy=[x,300,x+40,340],confidence=score,class_id=cls,source_tile=tile)
def case(rows):
    return dict(image='synthetic.JPG',source_sha256='same_source',weight_sha256='test',
        predictions=dict(source_shape=[2736,3648],merged_predictions=copy.deepcopy(rows),
            edge_kept_predictions=copy.deepcopy(rows)),zoom_evidence=[])
def fixtures(full=False):
    rows=[row(100+i*60,.99-i*.005) for i in range(10 if full else 5)]
    teacher=case(rows+[row(1200,.3)])
    student=case([row(1200,.9)]);student['predictions']['edge_kept_predictions']+=[row(1200,.85,tile=1)]
    if full:teacher['predictions']['edge_kept_predictions'] += [dict(r,source_tile=1) for r in rows]
    return teacher,student

class FusionTests(unittest.TestCase):
    def test_preserve_old_and_add_supported(self):
        teacher,student=fixtures();original=copy.deepcopy(teacher)
        result=merge(teacher,student)
        self.assertEqual(len(result['student_additions']),1);self.assertEqual(result['primary'],teacher['predictions']['merged_predictions'][:5]);self.assertEqual(teacher,original)
    def test_full_budget_preserves_every_old_cue(self):
        teacher,student=fixtures(True);result=merge(teacher,student)
        self.assertEqual(len(result['all_predictions']),10);self.assertEqual(result['student_additions'],[])
    def test_identity_mismatch(self):
        teacher,student=fixtures();student['source_sha256']='other'
        self.assertEqual(merge(teacher,student)['fallback_reason'],'source_identity_mismatch')
    def test_geometry_mismatch(self):
        teacher,student=fixtures();student['predictions']['source_shape']=[1,1]
        self.assertEqual(merge(teacher,student)['student_additions'],[])
    def test_missing_student_geometry_falls_back(self):
        teacher,student=fixtures();del student['predictions']['source_shape']
        self.assertEqual(merge(teacher,student)['student_additions'],[])
    def test_cross_class_support_not_accepted(self):
        teacher,student=fixtures();teacher['predictions']['merged_predictions'][-1]['class_id']=1
        self.assertEqual(merge(teacher,student)['student_additions'],[])
    def test_second_variant_does_not_require_teacher_support(self):
        teacher,student=fixtures();teacher['predictions']['merged_predictions']=teacher['predictions']['merged_predictions'][:5]
        self.assertEqual(len(merge(teacher,student,'student_consistent')['student_additions']),1)
    def test_single_tile_not_two_votes(self):
        teacher,student=fixtures();student['predictions']['edge_kept_predictions'][1]['source_tile']=0
        self.assertEqual(merge(teacher,student)['student_additions'],[])
    def test_existing_overlap_not_added(self):
        teacher,student=fixtures();student=case([row(100,.98)]);student['predictions']['edge_kept_predictions'] += [row(100,.9,tile=1)]
        self.assertEqual(merge(teacher,student)['student_additions'],[])
    def test_score_boundary_strict(self):
        teacher,student=fixtures();student['predictions']['merged_predictions'][0]['confidence']=.75
        self.assertEqual(merge(teacher,student)['student_additions'],[])

if __name__=='__main__':unittest.main()
