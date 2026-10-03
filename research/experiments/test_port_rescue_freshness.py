import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from inspection_agent.port_rescue_gui import snapshot,payload_is_current

class Value:
    def __init__(self,value):self.value=value
    def text(self):return self.value
    def currentData(self):return self.value
    def isChecked(self):return self.value

class FreshnessTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        for name in ('source','reference','aligned.jpg','report.json'):(root/name).write_bytes(b'original')
        report=dict(inspection=str(root/'source'),reference=str(root/'reference'),analysis_check_rois=[[0,0,1,1]])
        self.window=SimpleNamespace(_port_visual_report=report,current_output=root,_rescue_token='request',
            inspection=Value(report['inspection']),reference=Value(report['reference']),
            port_scene_combo=Value('scene'),rescue_switch=Value(True))
        self.payload=dict(token='request',snapshot=snapshot(self.window))
        self.assertTrue(payload_is_current(self.window,self.payload))
    def test_source_bytes(self):
        Path(self.window._port_visual_report['inspection']).write_bytes(b'changed')
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_reference_bytes(self):
        Path(self.window._port_visual_report['reference']).write_bytes(b'changed')
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_aligned_bytes(self):
        (self.window.current_output/'aligned.jpg').write_bytes(b'changed')
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_roi_changed(self):
        self.window._port_visual_report['analysis_check_rois']=[[.1,.1,.9,.9]]
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_scene_changed(self):
        self.window.port_scene_combo.value='other'
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_source_path_changed(self):
        self.window.inspection.value='other'
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_other_request(self):
        self.window._rescue_token='new'
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_disabled(self):
        self.window.rescue_switch.value=False
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_supplementary_switch_changed(self):
        self.window.rescue_supplement_switch=Value(True)
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_disk_report_changed(self):
        (self.window.current_output/'report.json').write_bytes(b'changed')
        self.assertFalse(payload_is_current(self.window,self.payload))

if __name__=='__main__':unittest.main()
