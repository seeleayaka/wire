import unittest
from unittest.mock import patch
from test_resolution_plug_budget_order import ClassBudgetOrderTests
from port_resolution_plug_policy_v3 import append_resolution_plugs_v3
class FinalClassBudgetTests(ClassBudgetOrderTests):
    def test_disallowed_jack_does_not_spend_last_free_slot(self):
        with patch('test_resolution_plug_budget_order.append_resolution_plugs_v2',append_resolution_plugs_v3):
            super().test_disallowed_jack_does_not_spend_last_free_slot()
if __name__=='__main__':unittest.main()
