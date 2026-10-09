import unittest
from copy import deepcopy
from human_bundle_recheck_bound import recheck
import test_human_bundle_recheck as fixtures

class ReferenceBindingTests(unittest.TestCase):
    def test_downloadable_report_contains_reference_photo_and_anchor_snapshot(self):
        scope, case, submission = fixtures.HumanRecheckTests().fixture()
        result = recheck(scope, case, submission)
        self.assertEqual(result['reference_binding'], scope['reference_binding'])
        self.assertEqual(result['reference_anchor_declarations'], scope['anchors'])
        self.assertEqual(result['reference_review'], scope['reference_review'])

    def test_reference_snapshot_is_independent_and_does_not_change_inputs(self):
        scope, case, submission = fixtures.HumanRecheckTests().fixture()
        before = deepcopy(scope)
        result = recheck(scope, case, submission)
        result['reference_binding']['image_sha256'] = 'changed'
        result['reference_anchor_declarations'][0]['id'] = 'changed'
        result['reference_review']['reviewer'] = 'changed'
        self.assertEqual(scope, before)

if __name__ == '__main__': unittest.main()
