from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from inspection_agent.workflow import InspectionTask,WorkflowError
from inspection_agent.terminal_mapping import create_mapping_draft
from inspection_agent.local_evidence_bridge import attach_review_file

class LocalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.image=self.root/'source.png';Image.new('RGB',(20,20),'white').save(self.image)
        mapping=create_mapping_draft(self.image,'fixture')
        mapping['ports']=[dict(id='p',device_id='draft',terminal_label='p',roi_kind='wire_entry_port',bbox_xyxy=[1,1,5,5],confirmed=False)]
        self.files={'case':{k:self.root/(k+'.json') for k in ('map','endpoints','audit')}}
        def dump(p,value):p.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
        self.dump=dump;dump(self.files['case']['map'],mapping);dump(self.files['case']['endpoints'],{})
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        audit=dict(image_binding=mapping['image_binding'],entry_map_sha256=sha(self.files['case']['map']),ports=[dict(port_id='p',candidates=[dict(record_id='r')])])
        dump(self.files['case']['audit'],audit)
        self.bundle=dict(schema_version=1,source_versions={'case':dict(image_binding=mapping['image_binding'],map_sha256=sha(self.files['case']['map']),endpoint_report_sha256=sha(self.files['case']['endpoints']),audit_sha256=sha(self.files['case']['audit']))},items=[dict(item_id='case:p',case='case',port_id='p',bbox_xyxy=[1,1,5,5],candidate_record_ids=['r'])])
        self.rehash_bundle();self.bp=self.root/'bundle.json';dump(self.bp,self.bundle)
        self.review=dict(schema_version=2,bundle_id=self.bundle['bundle_id'],source_versions=deepcopy(self.bundle['source_versions']),review_mode='automation_fixture',reviewer='TEST_ONLY',claim_scope='local_visible_relation_only',connection_edges=[],items=[dict(item_id='case:p',bbox_xyxy=[1,1,5,5],entry_review=dict(state='pending',evidence_note=''),candidate_reviews=[dict(record_id='r',state='supported',evidence_note='synthetic fixture only')])])
        self.rp=self.root/'review.json';dump(self.rp,self.review);self.tp=self.root/'task.json'
        task=InspectionTask('fixture','fixture',str(self.image),str(self.image));task.record_visual_analysis(dict(decision='possible_difference_manual_review',review_regions=[]));task.save(self.tp)
    def rehash_bundle(self):
        self.bundle['bundle_id']=hashlib.sha256(json.dumps({k:v for k,v in self.bundle.items() if k!='bundle_id'},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    def attach(self,allow=True):return attach_review_file(self.tp,self.bp,self.rp,self.files,'case',allow_test_records=allow)
    def test_default_rejects_fixture_without_writing(self):
        before=self.tp.read_bytes()
        with self.assertRaises(ValueError):self.attach(False)
        self.assertEqual(before,self.tp.read_bytes())
    def test_attach_persist_reload_no_human_verdict(self):
        result=self.attach();task=InspectionTask.load(self.tp);report=task.to_report()
        self.assertEqual(task.state,'awaiting_human_review');self.assertEqual(report['human_conclusions'],[]);self.assertEqual(report['topology_assessments'],[])
        self.assertEqual(result['connection_edges'],[]);self.assertFalse(result['human_fault_confirmation']);self.assertEqual(len(report['local_evidence_reviews']),1)
        self.assertFalse(list(self.root.glob('*.tmp')))
    def test_source_tampering_preserves_task(self):
        before=self.tp.read_bytes();self.files['case']['endpoints'].write_text('{} ',encoding='utf-8')
        with self.assertRaises(ValueError):self.attach()
        self.assertEqual(before,self.tp.read_bytes())
    def test_wrong_task_image_and_state(self):
        task=InspectionTask.load(self.tp);other=self.root/'other.png';Image.new('RGB',(20,20),'black').save(other)
        report=task.to_report();report['inputs']['inspection']=str(other);InspectionTask.from_report(report).save(self.tp)
        with self.assertRaises(ValueError):self.attach()
        report['inputs']['inspection']=str(self.image);report['state']='completed_no_actionable_difference';InspectionTask.from_report(report).save(self.tp)
        with self.assertRaises(ValueError):self.attach()
    def test_bundle_recomputed_with_unknown_candidate_rejected(self):
        self.bundle['items'][0]['candidate_record_ids']=['invented'];self.rehash_bundle();self.dump(self.bp,self.bundle)
        with self.assertRaises(ValueError):self.attach()
    def test_duplicate_preserves_file(self):
        self.attach();before=self.tp.read_bytes()
        with self.assertRaises(WorkflowError):self.attach()
        self.assertEqual(before,self.tp.read_bytes())
    def test_reinspection_attachment_distinct_and_cannot_add_guidance(self):
        first=self.attach();task=InspectionTask.load(self.tp)
        with self.assertRaises(WorkflowError):task.add_repair_guidance('unsupported',evidence_ids=['r'])
        task.record_human_review(reviewer='TEST',outcome='recapture_required',notes='synthetic test')
        task.start_reinspection(str(self.image));task.record_visual_analysis(dict(review_regions=[]));task.save(self.tp)
        second=self.attach();self.assertNotEqual(first['attachment_id'],second['attachment_id']);self.assertEqual(second['phase'],'reinspection')
if __name__=='__main__':unittest.main()
