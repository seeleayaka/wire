from __future__ import annotations

import unittest

from inspection_agent import (
    Image2NetAdapterError,
    compare_topologies,
    connection_graph_from_image2net,
)


def sample_netlist() -> dict:
    return {
        "ckt_type": "unit-test",
        "ckt_netlist": [
            {"component_type": "Res", "port_connection": {"Pos": "VDD", "Neg": "net1"}},
            {
                "component_type": "NMOS",
                "port_connection": {"Drain": "net1", "Gate": "VDD", "Source": "GND"},
            },
        ],
    }


class Image2NetAdapterTests(unittest.TestCase):
    def test_exact_netlist_maps_to_zero_distance(self) -> None:
        value = sample_netlist()
        expected = connection_graph_from_image2net(value, graph_id="expected")
        observed = connection_graph_from_image2net(value, graph_id="observed")
        result = compare_topologies(expected, observed)
        self.assertEqual(result["topology_distance"]["normalized_distance"], 0.0)
        self.assertEqual(result["decision"], "match")

    def test_one_rewired_port_is_detected(self) -> None:
        expected_value = sample_netlist()
        observed_value = sample_netlist()
        observed_value["ckt_netlist"][0]["port_connection"]["Pos"] = "GND"
        expected = connection_graph_from_image2net(expected_value, graph_id="expected")
        observed = connection_graph_from_image2net(observed_value, graph_id="observed")
        result = compare_topologies(expected, observed)
        self.assertEqual(result["summary"]["missing_count"], 1)
        self.assertEqual(result["summary"]["unexpected_count"], 1)
        self.assertGreater(result["topology_distance"]["normalized_distance"], 0.0)

    def test_missing_port_map_is_rejected(self) -> None:
        with self.assertRaises(Image2NetAdapterError):
            connection_graph_from_image2net(
                {"ckt_netlist": [{"component_type": "Res", "port_connection": {}}]},
                graph_id="bad",
            )


if __name__ == "__main__":
    unittest.main()
