import ast
from pathlib import Path
import unittest
from bundle_batch_contract import validate_preparation


class BatchContractTests(unittest.TestCase):
    def fixture(self):
        rows = [{'id': identity, 'phenotype': 'mating_body_visible', 'sam_inference_requested': True,
                 'crop_context': {'source': 'fixture'}, 'descriptor': [0], 'socket_patch_path': 'fixture'}
                for identity in ['reference', 'case_01', 'case_02']]
        rows[-1]['phenotype'] = 'uncertain'
        rows[-1]['sam_inference_requested'] = False
        return rows, [{'id': 'reference'}, {'id': 'case_01'}]

    def test_frozen_roster_accepts_unknown_without_sam(self):
        self.assertTrue(validate_preparation(*self.fixture(), inspection_count=2))

    def test_omitted_or_duplicate_inspection_rejected(self):
        for indices in [[0, 1], [0, 1, 1]]:
            rows, cases = self.fixture()
            with self.assertRaises(ValueError):
                validate_preparation([rows[i] for i in indices], cases, 2)

    def test_uncertain_sam_and_missing_pixels_rejected(self):
        rows, cases = self.fixture()
        rows[-1]['sam_inference_requested'] = True
        with self.assertRaises(ValueError):
            validate_preparation(rows, cases + [{'id': 'case_02'}], 2)
        rows, cases = self.fixture()
        del rows[1]['descriptor']
        with self.assertRaises(ValueError):
            validate_preparation(rows, cases, 2)

    def test_roster_mismatch_rejected(self):
        rows, cases = self.fixture()
        with self.assertRaises(ValueError):
            validate_preparation(rows, cases[:1], 2)

    def test_reference_not_ready_rejected(self):
        rows, cases = self.fixture()
        rows[0]['phenotype'] = 'uncertain'
        with self.assertRaises(ValueError):
            validate_preparation(rows, cases, 2)

    def test_all_new_batch_modules_parse_without_mutating_old_demo(self):
        folder = Path(__file__).parent
        for name in ['prepare_confirmed_bundle_batch', 'run_bundle_batch_geometry',
                     'analyze_confirmed_bundle_batch', 'audit_confirmed_bundle_batch',
                     'run_bundle_batch_geometry_v2', 'analyze_confirmed_bundle_batch_v2',
                     'audit_confirmed_bundle_batch_v2', 'stage_bundle_batch_runtime_v2',
                     'run_confirmed_bundle_batch_pipeline']:
            ast.parse((folder/(name+'.py')).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
