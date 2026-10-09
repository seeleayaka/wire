import unittest
from audit_sam_prompt_source_gate import assess


class SourceGateTests(unittest.TestCase):
    def make_rows(self):
        return [dict(id=i, prompt='cable', boxed=True, eligible=n, high_socket_touch=h)
                for i,n,h in [('reference',1,True),('source_visible_01',0,True),
                              ('source_visible_02',1,True),('source_exposed_01',0,False),
                              ('source_exposed_02',0,False)]]

    def test_baseline_and_ambiguity(self):
        rows = self.make_rows()
        self.assertEqual(assess(rows,'cable',True), (True,{'reference','source_visible_02'},set()))
        rows[0]['eligible'] = 2
        self.assertFalse(assess(rows,'cable',True)[0])

    def test_conflict_is_not_attachment(self):
        rows = self.make_rows()
        rows[3]['high_socket_touch'] = True
        rows[3]['eligible'] = 1
        _,visible,conflicts = assess(rows,'cable',True)
        self.assertNotIn('source_exposed_01', visible)
        self.assertEqual(conflicts, {'source_exposed_01'})

    def test_no_missing_or_duplicate_controls(self):
        rows = self.make_rows()
        with self.assertRaises(AssertionError): assess(rows[:-1],'cable',True)
        rows[-1]['id'] = rows[-2]['id']
        with self.assertRaises(AssertionError): assess(rows,'cable',True)


if __name__ == '__main__': unittest.main()
