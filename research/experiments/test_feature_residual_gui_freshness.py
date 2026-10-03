import unittest
from unittest.mock import patch
from test_port_rescue_freshness import FreshnessTests,Value
from inspection_agent.port_rescue_gui import snapshot,payload_is_current


class FeatureFreshnessTests(FreshnessTests):
    def enable_feature(self):
        self.window.rescue_supplement_switch=Value(True)
        self.window.rescue_student_switch=Value(True)
        self.window.rescue_feature_switch=Value(True)
        self.fingerprint=dict(feature='original',manifest='original',feature_code='original')
        patch('inspection_agent.port_rescue_gui.support_runtime_fingerprint',return_value={'student':'old'}).start()
        patch('inspection_agent.port_rescue_gui.residual_runtime_fingerprint',side_effect=lambda _:dict(self.fingerprint),create=True).start()
        self.addCleanup(patch.stopall)
        self.payload['snapshot']=snapshot(self.window)
        self.assertTrue(self.payload['snapshot']['feature'])
        self.assertTrue(payload_is_current(self.window,self.payload))
    def test_feature_weight_changed_rejects_stale(self):
        self.enable_feature();self.fingerprint['feature']='changed'
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_manifest_changed_rejects_stale(self):
        self.enable_feature();self.fingerprint['manifest']='changed'
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_feature_code_changed_rejects_stale(self):
        self.enable_feature();self.fingerprint['feature_code']='changed'
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_feature_switch_changed_rejects_stale(self):
        self.enable_feature();self.window.rescue_feature_switch.value=False
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_old_student_disabled_disables_feature(self):
        self.enable_feature();self.window.rescue_student_switch.value=False
        self.assertFalse(snapshot(self.window)['feature'])
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_supplement_disabled_disables_feature(self):
        self.enable_feature();self.window.rescue_supplement_switch.value=False
        self.assertFalse(snapshot(self.window)['feature'])
    def test_feature_off_no_new_reads(self):
        with patch('inspection_agent.port_rescue_gui.residual_runtime_fingerprint',side_effect=AssertionError('no new reads'),create=True):
            self.assertFalse(snapshot(self.window)['feature'])


if __name__=='__main__':unittest.main()
