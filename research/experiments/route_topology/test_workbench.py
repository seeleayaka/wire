import base64
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import build_mask_workbench as builder
from core import sha256
from test_core import mask
from test_pipeline import create_fixture


class Workbench(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.run,self.side=create_fixture(self.root,'fixture',mask([(20,30),(100,30)]))

    def tearDown(self):
        self.temp.cleanup()

    def test_embedded_source_and_mask_bytes_exactly_bound(self):
        page=builder.execute(self.run,self.run,self.root/'out')
        payload=json.loads(re.search(r'<script id="evidence" type="application/json">(.*?)</script>',page.read_text(encoding='utf-8'),re.S).group(1))
        for side in payload.values():
            original=base64.b64decode(side['data_url'].split(',',1)[1])
            self.assertEqual(hashlib.sha256(original).hexdigest(),side['image_binding']['image_sha256'])
            for item in side['masks']:
                pixels=base64.b64decode(item['data_url'].split(',',1)[1])
                self.assertEqual(hashlib.sha256(pixels).hexdigest(),item['sha256'])

    def test_embedded_fixture_kind_and_no_fresh_inference_claim(self):
        builder.execute(self.run,self.run,self.root/'out')
        report=json.loads((self.root/'out/report.json').read_text(encoding='utf-8'))
        self.assertFalse(report['fresh_SAM_inference'])
        self.assertEqual(report['new_confirmed_connections'],0)
        self.assertIn('constructed_software_fixture',report['origins']['reference']['origin_evidence_kind'])

    def test_source_read_race_not_silently_embedded(self):
        original_load=builder.load_run
        def drifting_load(directory):
            origin,records=original_load(directory)
            Image.fromarray(np.ones((120,120,3),np.uint8)).save(self.side['image_path'])
            return origin,records
        with patch.object(builder,'load_run',side_effect=drifting_load):
            with self.assertRaisesRegex(ValueError,'source bytes changed'):
                builder.execute(self.run,self.run,self.root/'out')
        self.assertFalse((self.root/'out').exists())

    def test_mask_read_race_not_silently_embedded(self):
        original_load=builder.load_run
        def drifting_load(directory):
            origin,records=original_load(directory)
            Image.fromarray(np.zeros((120,120),np.uint8)).save(self.run/'sam/mask_001.png')
            return origin,records
        with patch.object(builder,'load_run',side_effect=drifting_load):
            with self.assertRaisesRegex(ValueError,'mask bytes changed'):
                builder.execute(self.run,self.run,self.root/'out')
        self.assertFalse((self.root/'out').exists())

    def test_old_workbench_not_overwritten(self):
        page=builder.execute(self.run,self.run,self.root/'out');digest=sha256(page)
        with self.assertRaises(FileExistsError):builder.execute(self.run,self.run,self.root/'out')
        self.assertEqual(sha256(page),digest)

    def test_runner_and_template_fingerprints_recorded(self):
        builder.execute(self.run,self.run,self.root/'out')
        report=json.loads((self.root/'out/report.json').read_text(encoding='utf-8'))
        self.assertTrue(all(sha256(p)==digest for p,digest in report['source_pins'].items()))


if __name__=='__main__':
    unittest.main(verbosity=2)
