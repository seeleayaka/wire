import copy
import unittest
from paired_geometry_live_contract import validate_upstream


class UpstreamContract(unittest.TestCase):
    def setUp(self):
        self.report={'local_alignment':[{'dino_alignment_input':'local_ecc_corrected'}]}
        self.original=dict(status='fallback',fallback_reason='local_alignment_not_supported',
                           rescue_hints=[],supplementary_hints=[])

    def test_real_local_correction_abstains_without_rewriting_status(self):
        before=copy.deepcopy(self.original)
        self.assertEqual(validate_upstream(self.report,self.original,[],normal_control=True),
                         'safety_abstention_local_alignment')
        self.assertEqual(self.original,before)

    def test_model_failure_missing_provenance_candidate_or_hint_must_fail(self):
        variants=[({},self.original,[],True),(self.report,self.original,[{}],True),
                  (self.report,self.original,[],False),
                  (self.report,{**self.original,'fallback_reason':'model_failed'},[],True),
                  (self.report,{**self.original,'rescue_hints':[{}]},[],True)]
        for report,original,expected,normal in variants:
            with self.subTest(report=report,original=original,normal=normal):
                with self.assertRaises(AssertionError):
                    validate_upstream(report,original,expected,normal_control=normal)

if __name__=='__main__':unittest.main()
