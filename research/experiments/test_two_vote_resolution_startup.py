"""Import the real driver without starting inference or creating an output."""
import importlib
import unittest


class StartupTests(unittest.TestCase):
    def test_driver_uses_canonical_edge_filter(self):
        driver = importlib.import_module('run_two_vote_resolution')
        tiling = importlib.import_module('inspection_agent.port_tiling')
        self.assertIs(driver.near_artificial_edge, tiling.near_artificial_edge)


if __name__ == '__main__':
    unittest.main()

