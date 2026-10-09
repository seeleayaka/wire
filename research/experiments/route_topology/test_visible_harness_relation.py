import unittest
import numpy as np
import test_visible_bundle_relation as fixtures
from test_core import mask
from visible_harness_relation import observe_harness, compare_harness, POLICY


class HarnessRelationTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.BundleTests()
        fixture.setUp()
        self.scope,self.binding,self.poses,self.raw=fixture.scope,fixture.binding,fixture.poses,fixture.raw
        self.reference=self.observe(self.raw)

    def observe(self,raw,state='mating_body_visible',recipe=POLICY['recipe']):
        return observe_harness(self.scope,self.binding,self.poses,
            [{'raw':raw,'score':.9,'record_id':'r','recipe':recipe}],state,sam_inventory_verified=True)

    def test_distinct_recipe_and_one_observer(self):
        self.assertEqual(POLICY['recipe'],'wire_harness_plus_reference_anatomy_box')
        self.assertEqual(self.reference['kind'],'scoped_visible_whole_harness')
        self.assertEqual(self.reference['model_observer_count'],1)
        self.assertEqual(compare_harness(self.scope,self.reference,self.reference)['decision'],'same_visible_bundle_attachment_supported')

    def test_old_recipe_not_relabelled(self):
        old=self.observe(self.raw,recipe='cable_plus_reference_anatomy_box')
        self.assertFalse(old['unique_native_bundle_observation_supported'])

    def test_gap_still_abstains(self):
        raw=self.raw.copy();raw[:,50:70]=0
        self.assertEqual(compare_harness(self.scope,self.reference,self.observe(raw))['decision'],'insufficient_evidence')

    def test_socket_speck_conflict_not_overridden(self):
        raw=np.zeros_like(self.raw);raw[30,100]=255
        result=compare_harness(self.scope,self.reference,self.observe(raw,state='socket_contacts_exposed'))
        self.assertEqual(result['reason'],'exposed_socket_and_high_score_mask_conflict')
        self.assertEqual(result['decision'],'insufficient_evidence')

    def test_route_shape_not_a_connection_difference(self):
        raw=mask([(20,30),(20,70),(100,70),(100,30)])
        result=compare_harness(self.scope,self.reference,self.observe(raw))
        self.assertEqual(result['decision'],'same_visible_bundle_attachment_supported')
        self.assertEqual(result['electrical_correctness'],'not_assessed')


if __name__=='__main__':unittest.main()
