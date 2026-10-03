"""Verify review packet binding, original crop pixels and non-confirmation boundary."""
import sys,json,unittest
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,'E:/PythonProject10')
import numpy as np
from PIL import Image
from inspection_agent.optional_port_crop_review import sha
OUT=Path(__file__).resolve().parents[1]/'artifacts/port_endpoint_review_20261002'
PACKET=json.loads((OUT/'review_packet.json').read_text(encoding='utf-8'))
class PacketTests(unittest.TestCase):
    def test_existing_protected_files_unchanged(self):
        for path,value in PACKET['protected_files'].items():self.assertEqual(sha(path),value)
    def test_not_operator_review_or_edges(self):
        self.assertFalse(PACKET['operator_review']);self.assertEqual(PACKET['automatic_connections'],[])
        for case in PACKET['cases']:
            self.assertEqual(case['confirmed_connections'],[])
            for port in case['ports']:
                self.assertFalse(port['port_confirmed']);self.assertIsNone(port['assignment'])
                self.assertIsNone(port['human_conclusion']);self.assertIsNone(port['expected_connections'])
    def test_display_pixels_match_source_exactly(self):
        for case in PACKET['cases']:
            with Image.open(case['image_binding']['image_path']) as source:
                for port in case['ports']:
                    crop=source.convert('RGB').crop(tuple(port['context_source_xyxy']))
                    expected=crop.resize((crop.width*4,crop.height*4),Image.Resampling.NEAREST)
                    with Image.open(OUT/port['card']) as card:
                        self.assertTrue(np.array_equal(np.asarray(expected),np.asarray(card.crop((0,62,expected.width,62+expected.height)))))
    def test_comparison_score_separate_from_geometry(self):
        for case in PACKET['cases']:
            for port in case['ports']:
                for candidate in port['candidates']:
                    self.assertEqual(candidate['meets_existing_comparison_score'],candidate['source_score']>=.75)
                    self.assertIsNone(candidate['actual_terminal_identity']);self.assertIsNone(candidate['wire_label_reading'])
    def test_formal_topology_still_abstains(self):
        for case in PACKET['cases']:
            result=json.loads((OUT/case['formal_assessment_file']).read_text(encoding='utf-8'))
            self.assertEqual(result['decision'],'insufficient_evidence');self.assertIsNone(result['observed_graph'])
            self.assertIsNone(result['raw_comparison']);self.assertEqual(result['confirmed_port_count'],0)
if __name__=='__main__':unittest.main()
