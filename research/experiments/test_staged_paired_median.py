"""Run actual portable contracts before installation without changing E."""
import importlib.util
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; REPO = Path('E:/PythonProject10')
sys.path[:0] = [str(REPO), str(REPO / 'prototype')]
STAGING = ROOT / 'staging/paired_median_release'


def load_tests(loader, tests, pattern):
    import inspection_agent
    names = ('inspection_agent.paired_port_median_features', 'inspection_agent.paired_median_geometry')
    old_modules = {name: sys.modules.get(name) for name in names}
    old_attribute = getattr(inspection_agent, 'paired_median_geometry', None)
    try:
        for name, file in zip(names, ('paired_port_median_features.py', 'paired_median_geometry.py')):
            spec = importlib.util.spec_from_file_location(name, STAGING / file)
            module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
        inspection_agent.paired_median_geometry = sys.modules[names[1]]
        spec = importlib.util.spec_from_file_location('median_staged_contract_tests', STAGING / 'test_paired_median_geometry.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return loader.loadTestsFromModule(module)
    finally:
        for name, module in old_modules.items():
            if module is None: sys.modules.pop(name, None)
            else: sys.modules[name] = module
        if old_attribute is None: del inspection_agent.paired_median_geometry
        else: inspection_agent.paired_median_geometry = old_attribute


if __name__ == '__main__': unittest.main()
