import unittest
from unittest.mock import patch
from test_port_rescue_freshness import FreshnessTests,Value
from inspection_agent.port_rescue_gui import snapshot,payload_is_current,STUDENT_POLICY_ID

class StudentFreshnessTests(FreshnessTests):
    def enable_student(self):
        self.window.rescue_supplement_switch=Value(True);self.window.rescue_student_switch=Value(True)
        self.fingerprint=dict(student='original',teacher='original',manifest='original')
        self.mock=patch('inspection_agent.port_rescue_gui.support_runtime_fingerprint',side_effect=lambda _:dict(self.fingerprint)).start()
        self.addCleanup(patch.stopall)
        self.payload['snapshot']=snapshot(self.window)
        self.assertEqual(self.payload['snapshot']['policy'],STUDENT_POLICY_ID)
        self.assertTrue(payload_is_current(self.window,self.payload))
    def test_student_model_changed(self):
        self.enable_student();self.fingerprint['student']='changed'
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_manifest_changed(self):
        self.enable_student();self.fingerprint['manifest']='changed'
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_student_switch_changed(self):
        self.enable_student();self.window.rescue_student_switch.value=False
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_supplementary_disables_student(self):
        self.enable_student();self.window.rescue_supplement_switch.value=False
        self.assertFalse(snapshot(self.window)['student'])
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_off_never_reads_new_models(self):
        with patch('inspection_agent.port_rescue_gui.support_runtime_fingerprint',side_effect=AssertionError('must not read')):
            self.assertFalse(snapshot(self.window)['student'])
if __name__=='__main__':unittest.main()
