import unittest
from test_feature_residual_gui_freshness import FeatureFreshnessTests
from inspection_agent.port_rescue_gui import snapshot,payload_is_current
class ResolutionFreshnessTests(FeatureFreshnessTests):
    def enable_feature(self):
        super().enable_feature()
        self.fingerprint.update(resolution_code='old',resolution_manifest='old',predict_code='old',tiling_code='old')
        self.payload['snapshot']=snapshot(self.window)
    def test_resolution_manifest_rejects_stale(self):
        self.enable_feature();self.fingerprint['resolution_manifest']='changed'
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_resolution_code_rejects_stale(self):
        self.enable_feature();self.fingerprint['resolution_code']='changed'
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_prediction_code_rejects_stale(self):
        self.enable_feature();self.fingerprint['predict_code']='changed'
        self.assertFalse(payload_is_current(self.window,self.payload))
    def test_tiling_code_rejects_stale(self):
        self.enable_feature();self.fingerprint['tiling_code']='changed'
        self.assertFalse(payload_is_current(self.window,self.payload))
if __name__=='__main__':unittest.main()
