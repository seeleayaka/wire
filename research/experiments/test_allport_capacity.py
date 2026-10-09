import unittest
from audit_allport_capacity import budget_ceiling


class CapacityTests(unittest.TestCase):
    def test_ideal_extra_slots_not_achieved_hits(self):
        r=budget_ceiling(12,8,8,5)
        self.assertEqual(r['ideal_new_hits_upper_bound'],2);self.assertEqual(r['unavoidable_misses_lower_bound'],2)

    def test_existing_unmatched_rows_consume_budget(self):
        r=budget_ceiling(10,8,10,5)
        self.assertEqual(r['ideal_new_hits_upper_bound'],0);self.assertEqual(r['unavoidable_misses_lower_bound'],2)

    def test_fewer_primary_is_not_automatic_top10_budget(self):
        self.assertEqual(budget_ceiling(10,6,6,2)['capacity'],7)

    def test_normal_zero_targets_not_normal_success_metric(self):
        self.assertEqual(budget_ceiling(0,0,0,0)['ideal_new_hits_upper_bound'],0)

    def test_invalid_count_contract_rejected(self):
        for args in [(12,8,11,5),(8,9,9,5),(10,6,5,5),(8,5,5,6),(0.,0,0,0),(True,0,0,0)]:
            with self.subTest(args=args),self.assertRaises(ValueError):budget_ceiling(*args)


if __name__=='__main__':unittest.main(verbosity=2)
