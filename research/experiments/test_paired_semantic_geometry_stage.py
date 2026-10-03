import types
import unittest
from unittest.mock import patch
import run_paired_semantic_geometry_stage as adapter


class AdapterTests(unittest.TestCase):
    def test_enriched_out_fold_derivation_and_restoration(self):
        source=dict(source_groups=['c','a','b'],sources=[dict(image='a'),dict(image='c'),dict(image='b')])
        original_load=lambda path:source
        fake=types.SimpleNamespace(OUT='old',load=original_load)
        def main():
            self.assertEqual(fake.OUT,adapter.NEW)
            derived=fake.load(adapter.NEW/'features_train/report.json')
            self.assertEqual([r['fold'] for r in derived['sources']],[0,2,1])
            self.assertTrue(all('fold' not in r for r in source['sources']))
        fake.main=main
        with patch.object(adapter.importlib,'import_module',return_value=fake):adapter.execute('source')
        self.assertEqual(fake.OUT,'old');self.assertIs(fake.load,original_load)

    def test_failure_always_restores_original_modules(self):
        original_load=lambda path:None
        def fail():raise RuntimeError('expected')
        fake=types.SimpleNamespace(OUT='old',load=original_load,main=fail)
        with patch.object(adapter.importlib,'import_module',return_value=fake):
            with self.assertRaises(RuntimeError):adapter.execute('classifier')
        self.assertEqual(fake.OUT,'old');self.assertIs(fake.load,original_load)


if __name__=='__main__':unittest.main()
