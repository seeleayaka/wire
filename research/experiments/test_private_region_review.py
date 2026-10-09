"""Local preparation, preview and request-scope tests; no real API or SAM."""
import copy,json,base64,unittest
from pathlib import Path
import launch_llm_recheck_window_20261008 as window
import private_region_review as private
import llm_review_priority as priority
from PyQt5.QtWidgets import QApplication,QMessageBox
from PyQt5.QtCore import QRect,QPoint,Qt
from PyQt5.QtTest import QTest
from unittest.mock import patch
from probe_llm_recheck_planner_20261008 import CASES,ROOT

class PrivateReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        report=json.loads(CASES['cabinet2'].read_text(encoding='utf-8'));f=report['sam3_fusion']
        self.pending={'report':report,'output':CASES['cabinet2'].parent,'candidates':report['review_regions'],
                      'reference_mask':Path(f['reference_sam3']['output_dir'])/'mask_union.png',
                      'inspection_mask':Path(f['inspection_sam3']['output_dir'])/'mask_union.png'}
    def packet(self):
        prepared=private.prepare(self.pending);dialog=private.PreviewDialog(prepared)
        for pair in dialog.canvases:
            for canvas in pair:canvas.redactions=[canvas.original.rect()]
        dialog.make_preview();dialog.approval.setEnabled(True);dialog.approval.setChecked(True)
        dialog.accept();packet=dialog.packet;dialog.close();return packet
    def test_real_crop_preparation_preserves_candidates(self):
        frozen=copy.deepcopy(self.pending);result=private.prepare(self.pending)
        self.assertEqual(len(result['regions']),2);self.assertEqual(self.pending,frozen)
        for r in result['regions']:self.assertLess(r['images'][0].width(),575)
    def test_preview_cannot_accept_without_operator(self):
        d=private.PreviewDialog(private.prepare(self.pending));d.accept()
        self.assertIsNone(d.packet);d.close()
    def test_solid_redaction_removes_pixels(self):
        prepared=private.prepare(self.pending);im=prepared['regions'][0]['images'][0]
        c=private.RedactionCanvas(im,[im.rect()]);redacted=c.render()
        self.assertEqual(redacted.pixelColor(5,5).name(),'#20252b')
        c.undo();self.assertEqual(private.png(c.render()),private.png(im))
    def test_redaction_edit_invalidates_exact_preview(self):
        d=private.PreviewDialog(private.prepare(self.pending));d.make_preview()
        d.approval.setEnabled(True);d.approval.setChecked(True);d.canvases[0][0].undo()
        self.assertIsNone(d.preview_rows);self.assertFalse(d.approval.isChecked());d.accept()
        self.assertIsNone(d.packet);d.close()
    def test_style_switch_invalidates_preview(self):
        d=private.PreviewDialog(private.prepare(self.pending));d.make_preview()
        d.approval.setEnabled(True);d.approval.setChecked(True)
        d.style_selector.setCurrentIndex(2)
        self.assertIsNone(d.preview_rows);self.assertFalse(d.approval.isChecked())
        self.assertFalse(d.approval.isEnabled());d.close()
    def test_small_mosaic_preserves_outside_pixels(self):
        im=private.prepare(self.pending)['regions'][0]['images'][0]
        before=private.png(im);rect=QRect(3,3,12,12)
        c=private.RedactionCanvas(im,[rect]);c.set_privacy_style('mosaic',6);out=c.render()
        self.assertEqual(before,private.png(im))
        for y in range(im.height()):
            for x in range(im.width()):
                if not rect.contains(x,y):self.assertEqual(out.pixelColor(x,y),im.pixelColor(x,y))
    def test_zoomed_drag_maps_to_original_pixels(self):
        prepared=private.prepare(self.pending);im=prepared['regions'][0]['images'][0]
        c=private.RedactionCanvas(im,[]);c.show()
        QTest.mousePress(c,Qt.LeftButton,pos=QPoint(round(5*c.zoom),round(5*c.zoom)))
        QTest.mouseRelease(c,Qt.LeftButton,pos=QPoint(round(20*c.zoom),round(20*c.zoom)))
        self.assertEqual(c.render().pixelColor(10,10).name(),'#20252b')
        self.assertNotEqual(c.render().pixelColor(30,30).name(),'#20252b');c.close()
    def test_masks_are_not_redacted_or_reconstructed(self):
        prepared=private.prepare(self.pending);r=prepared['regions'][0]
        before=[private.png(x) for x in r['images'][2:]]
        private.panel(r,r['images'][0],r['images'][1])
        self.assertEqual(before,[private.png(x) for x in r['images'][2:]])
    def test_full_frame_photo_export_is_rejected(self):
        broad=copy.deepcopy(self.pending)
        with private.Image.open(broad['reference_mask']) as im:w,h=im.size
        broad['candidates']=[{'bbox_xyxy':[0,0,w,h]}]
        with self.assertRaises(ValueError):private.prepare(broad)
    def test_packet_integrity(self):
        packet=self.packet();self.assertEqual(private.validate(packet,self.pending),['candidate_001','candidate_002'])
        packet['regions'][0]['png_base64']='bad'
        with self.assertRaises(ValueError):private.validate(packet,self.pending)
    def test_changed_input_rejects_preview(self):
        packet=self.packet();changed=copy.deepcopy(self.pending);changed['candidates'][0]['left']=99
        with self.assertRaises(ValueError):private.validate(packet,changed)
    def test_metadata_and_scope(self):
        packet=self.packet();request,ids=private.request_payload(packet,self.pending['reference_mask'],self.pending['inspection_mask'],self.pending['candidates'],priority.base.backend.load_settings())
        content=request['messages'][1]['content'];self.assertEqual(len(content),3)
        encoded=json.dumps(request)
        self.assertNotIn('HUAWEI',encoded);self.assertNotIn('bbox_normalized_xyxy',encoded)
        self.assertIn('PRIVACY REDACTIONS',encoded)
        for row in packet['regions']:
            raw=base64.b64decode(row['png_base64']);self.assertNotIn(b'eXIf',raw);self.assertNotIn(b'tEXt',raw)
    def test_no_photo_without_approval(self):
        packet=self.packet();packet['operator_reviewed']=False
        sender=[]
        result=priority.run(self.pending['reference_mask'],self.pending['inspection_mask'],self.pending['candidates'],api_key='fake',sender=lambda *_:sender.append(1),visual_packet=packet)
        self.assertEqual(result['network_requests'],0);self.assertEqual(sender,[])
    def test_photo_payload_with_mock_transport(self):
        packet=self.packet();fixture=json.loads((ROOT/'artifacts/llm_priority_cabinet2_diagnostic_20261008/response_fixture.json').read_text(encoding='utf-8'))
        requests=[]
        def sender(req,*_):
            requests.append(json.loads(req.data));return {'model':'mock','choices':[{'finish_reason':'stop','message':{'content':json.dumps(fixture)}}]}
        result=priority.run(self.pending['reference_mask'],self.pending['inspection_mask'],self.pending['candidates'],api_key='fake',sender=sender,visual_packet=packet)
        self.assertEqual(result['status'],'ok');self.assertEqual(result['input_mode'],'redacted_local_photos')
        self.assertEqual(len(requests),1);self.assertNotIn('png_base64',json.dumps(result))
    def test_window_default_and_switch(self):
        w=window.PlannedWindow();self.assertEqual(w.input_mode.currentData(),'binary_masks')
        self.assertFalse(w.preview_button.isEnabled());w._approved_visual_packet=self.packet()
        w.input_mode.setCurrentIndex(1);self.assertIsNone(w._approved_visual_packet);self.assertTrue(w.preview_button.isEnabled());w.close()
    def test_cancelled_send_never_starts_worker(self):
        w=window.PlannedWindow();w._deepseek_pending=self.pending;w.current_output=self.pending['output']
        with patch.object(QMessageBox,'question',return_value=QMessageBox.No):w.start_deepseek_mask_review()
        self.assertIsNone(w.deepseek_worker);w.close()
    def test_stale_photo_result_does_not_adjust(self):
        w=window.PlannedWindow();w._deepseek_pending=self.pending;w.current_output=self.pending['output'];saved=[]
        w._write_report=lambda x:saved.append(copy.deepcopy(x))
        w._approved_visual_packet=self.packet()
        fixture=json.loads((ROOT/'artifacts/llm_priority_cabinet2_diagnostic_20261008/result.json').read_text(encoding='utf-8'))
        fixture.update(input_mode='redacted_local_photos',visual_packet_sha256='wrong',visual_source_binding={})
        w._finish_deepseek_mask_review(fixture)
        self.assertEqual(saved[-1]['external_mask_review']['status'],'error')
        self.assertEqual(w.priority_table.item(0,3).text(),'普通待复核');w.close()

if __name__=='__main__':unittest.main()
