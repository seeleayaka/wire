from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from core import image_binding
from test_core import mask
from test_pipeline import create_fixture,save
from run_paired_evidence import load_run,lift_verified_context,execute
from paired_evidence import propose_correspondences


class ContextBinding(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.run,self.side=create_fixture(self.root,'crop',mask([(20,30),(100,30)]))
        self.origin,self.records=load_run(self.run)
        source=np.zeros((240,240,3),np.uint8)
        with Image.open(self.side['image_path']) as opened:
            source[40:160,50:170]=np.asarray(opened.convert('RGB'))
        self.source=self.root/'full.png';Image.fromarray(source).save(self.source)
        self.entry={'fresh_run_directory':str(self.run),
                    'source_binding':{'image_path':str(self.source),**image_binding(self.source)},
                    'crop_binding':{'image_path':self.side['image_path'],**self.origin['image_binding']},
                    'crop_box_xyxy':[50,40,170,160]}
        self.protocol=self.root/'context.json';save(self.protocol,{'cases':[self.entry]})

    def tearDown(self):
        self.temp.cleanup()

    def test_paths_separate_from_fingerprint(self):
        self.assertEqual(self.origin['image_path'],str(Path(self.side['image_path']).resolve()))
        self.assertNotIn('image_path',self.origin['image_binding'])

    def test_exact_translation_does_not_change_mask_identity(self):
        origin,records=lift_verified_context(self.origin,self.records,self.protocol)
        self.assertEqual(origin['coordinate_translation'],[50,40])
        self.assertEqual(records[0]['geometry']['tips_xy'],[[70,70],[150,70]])
        self.assertEqual(records[0]['mask_array_sha256'],self.records[0]['mask_array_sha256'])

    def test_original_crop_guards_not_cleared(self):
        records=deepcopy(self.records);records[0]['boundary_truncated']=True
        _,lifted=lift_verified_context(self.origin,records,self.protocol)
        self.assertTrue(lifted[0]['boundary_truncated'])

    def test_source_hash_drift_rejected(self):
        Image.fromarray(np.ones((240,240,3),np.uint8)).save(self.source)
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            lift_verified_context(self.origin,self.records,self.protocol)

    def test_correct_hash_but_wrong_region_rejected(self):
        self.entry['crop_box_xyxy']=[60,40,180,160]
        save(self.protocol,{'cases':[self.entry]})
        with self.assertRaisesRegex(ValueError,'pixels differ'):
            lift_verified_context(self.origin,self.records,self.protocol)

    def test_duplicate_context_run_rejected(self):
        save(self.protocol,{'cases':[self.entry,self.entry]})
        with self.assertRaisesRegex(ValueError,'uniquely'):
            lift_verified_context(self.origin,self.records,self.protocol)

    def test_crop_identity_drift_rejected(self):
        self.entry['crop_binding']['image_sha256']='0'*64;save(self.protocol,{'cases':[self.entry]})
        with self.assertRaisesRegex(ValueError,'crop fingerprint'):
            lift_verified_context(self.origin,self.records,self.protocol)

    def test_invalid_crop_coordinate_type_rejected(self):
        self.entry['crop_box_xyxy'][0]=True;save(self.protocol,{'cases':[self.entry]})
        with self.assertRaisesRegex(ValueError,'coordinates'):
            lift_verified_context(self.origin,self.records,self.protocol)

    def test_lift_does_not_mutate_input(self):
        origin,records=deepcopy(self.origin),deepcopy(self.records)
        lift_verified_context(self.origin,self.records,self.protocol)
        self.assertEqual(self.origin,origin);self.assertEqual(self.records,records)

    def test_driver_self_control_not_electrical_success(self):
        result=execute(self.run,self.run,self.root/'out')
        self.assertEqual(len(result['proposals']),1)
        self.assertEqual(result['topology_decision'],'insufficient_evidence')
        self.assertEqual(result['new_confirmed_connections'],0)
        self.assertIn('constructed_software_fixture',result['reference']['origin_evidence_kind'])

    def test_driver_preserves_old_output(self):
        execute(self.run,self.run,self.root/'out')
        with self.assertRaises(FileExistsError):
            execute(self.run,self.run,self.root/'out')

    def test_driver_context_keeps_crop_and_full_frames_explicit(self):
        r=execute(self.run,self.run,self.root/'context_out',self.protocol)
        self.assertEqual(r['reference']['image_binding']['image_size'],[120,120])
        self.assertEqual(r['reference']['frame_binding']['image_size'],[240,240])
        self.assertEqual(len(r['proposals']),1)


class DefensivePairing(unittest.TestCase):
    def setUp(self):
        from core import extract_mask
        self.record=extract_mask(mask([(20,30),(100,30)]),.9,'a')

    def pair(self,records=None,size=None,h=None):
        return propose_correspondences([self.record] if records is None else records,
                                       [self.record],np.eye(3) if h is None else h,
                                       {'reliable':True},[120,120] if size is None else size)

    def test_duplicate_record_id_rejected(self):
        with self.assertRaisesRegex(ValueError,'unique'):
            self.pair([self.record,self.record])

    def test_nonfinite_tip_rejected(self):
        record=deepcopy(self.record);record['geometry']['tips_xy'][0][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):
            self.pair([record])

    def test_invented_tips_not_used(self):
        record=deepcopy(self.record);record['geometry']['tips_xy'][0]=[25,30]
        with self.assertRaisesRegex(ValueError,'actual two tips'):
            self.pair([record])

    def test_nonfinite_frame_size_rejected(self):
        with self.assertRaises(ValueError):self.pair(size=[float('nan'),120])

    def test_corrupt_empty_frame_not_passed(self):
        with self.assertRaises(ValueError):self.pair(records=[],h=np.zeros((3,3)))

    def test_same_frame_all_records_insufficient_when_no_unique_pair(self):
        r=self.pair(records=[])
        self.assertEqual(r['proposals'],[])
        self.assertFalse(r['automatic_fault_verdict'])


if __name__=='__main__':
    unittest.main(verbosity=2)
