import copy
import unittest
from test_teacher_student_port_policy import fixtures,case,row
from teacher_student_port_policy import merge
from evaluate_full_backbone_acceptance import append_backbone


class CurrentBackbonePreservationTests(unittest.TestCase):
    def test_new_resolution_cue_is_part_of_protected_prefix(self):
        teacher,new=fixtures();current=merge(teacher,case([]))
        cue=row(700,.98);cue.update(resolution_policy_id='v3',loose_plug_only=True)
        current['resolution_additions']=[copy.deepcopy(cue)];current['all_predictions'].append(cue)
        old=copy.deepcopy(current);result=append_backbone(teacher,current,new)
        self.assertEqual(result['all_predictions'][:len(current['all_predictions'])],old['all_predictions'])
        self.assertEqual(result['resolution_additions'],old['resolution_additions'])
        self.assertEqual(current,old)
    def test_current_fifth_extra_blocks_backbone(self):
        teacher,new=fixtures();current=merge(teacher,case([]))
        current['all_predictions'] += [row(700+i*60,.98) for i in range(5)]
        self.assertEqual(append_backbone(teacher,current,new)['backbone_additions'],[])
    def test_current_duplicate_does_not_consume_second_slot(self):
        teacher,new=fixtures();current=merge(teacher,new)
        current['resolution_additions']=[dict(marker='must stay')]
        result=append_backbone(teacher,current,new)
        self.assertEqual(result['backbone_additions'],[])
        self.assertEqual(result['resolution_additions'],current['resolution_additions'])


if __name__=='__main__':unittest.main()
