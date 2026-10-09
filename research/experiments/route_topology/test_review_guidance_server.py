import json
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError
from PIL import Image
from core import sha256
from serve_review_guidance import companion_handler
from test_human_bundle_server import HumanServerTests


class GuidanceServerTests(HumanServerTests):
    def setUp(self):
        super().setUp()
        port = self.server.server_address[1]
        self.server.RequestHandlerClass = companion_handler(self.catalog, self.out, 'fixture_token', port)

    def test_guidance_and_page_are_available_without_model_calls(self):
        with urlopen(self.url + '/guidance?case=fixture') as r:
            plan = json.load(r)
        self.assertEqual(plan['model_calls'], 0)
        self.assertEqual(plan['network_calls'], 0)
        self.assertEqual(plan['automatic_comparison_unchanged'], self.catalog['cases'][0]['automatic_comparison'])
        with urlopen(self.url) as r:
            page = r.read().decode('utf-8')
        self.assertIn('id="guidance"', page)
        with self.assertRaises(HTTPError): urlopen(self.url + '/guidance?case=missing')

    def test_sidecar_is_bound_to_saved_original_and_not_a_verdict(self):
        status, result = self.post(self.s)
        self.assertEqual(status, 200)
        original = json.loads(Path(result['saved_report']).read_text(encoding='utf-8'))
        self.assertEqual(original, result['report'])
        self.assertNotIn('review_plan', original)
        plan = json.loads(Path(result['saved_review_plan']).read_text(encoding='utf-8'))
        self.assertEqual(plan, result['review_plan'])
        self.assertEqual(plan['review_report_sha256'], sha256(result['saved_report']))
        self.assertEqual(plan['review_source'], 'software_fixture')
        self.assertEqual(plan['automatic_new_hits'], 0)
        self.assertIn('非真实验收', Path(result['saved_review_plan']).with_suffix('.md').read_text(encoding='utf-8'))

    def test_input_evidence_drift_blocks_readonly_guidance(self):
        (self.out/'source_report.json').write_text('changed', encoding='utf-8')
        with self.assertRaises(HTTPError): urlopen(self.url + '/guidance?case=fixture')

    def test_photo_drift_blocks_guidance(self):
        Image.new('RGB', (200, 100), 'black').save(self.out/'images/fixture.jpg')
        with self.assertRaises(HTTPError): urlopen(self.url + '/guidance?case=fixture')
