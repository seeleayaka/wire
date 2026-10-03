from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

import dino_feature_diff as dino  # noqa: E402


class DinoCachePathTests(unittest.TestCase):
    def test_default_cache_is_user_temp_not_project_output(self) -> None:
        expected = Path(tempfile.gettempdir()) / "PythonProject10" / "dino_cache"
        self.assertEqual(dino.CACHE_DIR, expected)


if __name__ == "__main__":
    unittest.main()
