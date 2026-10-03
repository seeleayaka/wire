"""Load isolated staged source under its target package, then run tests."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, "E:/PythonProject10")
source = Path(__file__).with_name("terminal_mapping.py")
spec = importlib.util.spec_from_file_location("inspection_agent.terminal_mapping", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
test_spec = importlib.util.spec_from_file_location("terminal_mapping_tests_staged", source.with_name("test_terminal_mapping.py"))
test_module = importlib.util.module_from_spec(test_spec)
test_spec.loader.exec_module(test_module)
suite = unittest.defaultTestLoader.loadTestsFromModule(test_module)
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    raise SystemExit(1)
