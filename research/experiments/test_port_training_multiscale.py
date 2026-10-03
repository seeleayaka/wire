import copy,sys,unittest
sys.path.insert(0,'E:/PythonProject10')
from port_training_multiscale_policy import selected_seeds,labels_for_window,positive_window,hard_negative_selection

def port(i=0,cls=0,box=None):return dict(source_box_index=i,class_id=cls,box_xyxy=box or [1000,1000,1080,1060])
class SamplePolicyTests(unittest.TestCase):
    def test_one_seed_per_class_and_order_independent(self):
        rows=[port(0),port(1),port(2,1),port(3,1)]
        self.assertEqual(selected_seeds('source',rows),selected_seeds('source',rows[::-1]))
        self.assertEqual(len(selected_seeds('source',rows)),2)
    def test_partial_label_refuses_entire_crop(self):
        self.assertIsNone(labels_for_window([port(box=[500,500,750,550])],[600,400,1240,1040],[2736,3648]))
    def test_artificial_edge_refuses(self):
        self.assertIsNone(labels_for_window([port(box=[605,500,650,550])],[600,400,1240,1040],[2736,3648]))
    def test_native_frame_boundary_keeps_annotation(self):
        self.assertIsNotNone(labels_for_window([port(box=[0,100,50,150])],[0,0,640,640],[2736,3648]))
    def test_complete_other_targets_included(self):
        labels=labels_for_window([port(),port(1,1,[1100,1050,1130,1100])],[800,800,1440,1440],[2736,3648])
        self.assertEqual(len(labels),2)
    def test_positive_window_seed_kept_and_no_mutation(self):
        rows=[port(),port(1,1,[1100,1050,1130,1100])];before=copy.deepcopy(rows)
        result=positive_window(rows[0],rows,640,[2736,3648])
        self.assertIn(0,[p['source_box_index'] for p in result['labels']]);self.assertEqual(rows,before)
    def test_hard_negative_bounded_diversity(self):
        rows=[dict(source_image=f'{kind}_{i:03}.JPG',tile_id=0,teacher_score=.5+i*.001) for kind in ('normal','damaged','misrouted','disconnected') for i in range(35)]
        selected=hard_negative_selection(rows)
        self.assertEqual(len(selected),64);self.assertEqual(sum(r['source_image'].startswith('normal_') for r in selected),24)
    def test_hard_negative_one_per_source(self):
        rows=[dict(source_image='normal_001.JPG',tile_id=i,teacher_score=.5+i*.01) for i in range(10)]
        selected=hard_negative_selection(rows);self.assertEqual(len(selected),1);self.assertEqual(selected[0]['tile_id'],9)
    def test_no_port_not_positive(self):self.assertEqual(selected_seeds('normal_001.JPG',[]),[])

if __name__=='__main__':unittest.main()
