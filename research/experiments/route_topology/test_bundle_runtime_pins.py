import unittest
from unittest.mock import patch
from bundle_runtime_pins import source_pins


class RuntimePinsTests(unittest.TestCase):
    def test_stdlib_runtime_matches_canonical_readonly_E_snapshot(self):
        from run_audit import source_pins as canonical
        self.assertEqual(source_pins(), canonical())

    def test_dirty_tree_status_included_in_fingerprint(self):
        with patch('bundle_runtime_pins.subprocess.check_output', return_value=' M one.py\n'):
            first = source_pins()
        with patch('bundle_runtime_pins.subprocess.check_output', return_value=' M two.py\n'):
            second = source_pins()
        self.assertNotEqual(first['dirty_git_status_sha256'], second['dirty_git_status_sha256'])
        self.assertEqual({k: v for k, v in first.items() if k != 'dirty_git_status_sha256'},
                         {k: v for k, v in second.items() if k != 'dirty_git_status_sha256'})


if __name__ == '__main__':
    unittest.main()
