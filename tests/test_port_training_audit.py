import unittest
from inspection_agent.port_training_audit import parse_boxes,rectangle_labels,inner_split,plan_tiles


class AuditTests(unittest.TestCase):
    def test_source_mapping_and_polygon_roundtrip(self):
        boxes=parse_boxes('3 .5 .5 .2 .2\n4 .25 .25 .1 .1\n1 .2 .2 .1 .1',100,100)
        self.assertEqual(boxes[0]['box_xyxy'],[40.,40.,60.,60.])
        self.assertEqual(rectangle_labels(boxes)[0],'0 0.400000 0.400000 0.600000 0.400000 0.600000 0.600000 0.400000 0.600000')
        self.assertEqual(len(rectangle_labels(boxes)),2)

    def test_invalid_labels_fail_closed(self):
        for text in ['3.1 .5 .5 .1 .1','9 .5 .5 .1 .1','3 nan .5 .1 .1','3 .5 .5 0 .1','3 2 .5 .1 .1','3 .5 .5 .1']:
            with self.assertRaises(ValueError):parse_boxes(text,100,100)

    def test_clipped_valid_boxes_are_flagged(self):
        boxes=parse_boxes('3 .02 .02 .1 .1',100,100)
        self.assertTrue(boxes[0]['clipped'])
        self.assertEqual(boxes[0]['box_xyxy'],[0.,0.,7.000000000000001,7.000000000000001])

    def test_split_is_source_grouped_and_order_stable(self):
        names=[f'{kind}_{i:03}.JPG' for kind in ('damaged','disconnected','misrouted','normal') for i in range(10)]
        split=inner_split(names)
        self.assertEqual(split,inner_split(list(reversed(names))))
        self.assertEqual(sum(v=='inner_val' for v in split.values()),8)
        with self.assertRaises(ValueError):inner_split(names+[names[0]])

    def test_cut_port_excludes_entire_tile_not_just_label(self):
        boxes=parse_boxes('3 .5 .5 .1 .1',1800,1280)
        # x=0 and end anchor x=520 both fully contain this port.
        self.assertTrue(all(t['usable'] for t in plan_tiles(boxes,1800,1280)))
        cut=parse_boxes('3 .7 .5 .1 .1',1800,1280)
        tiles=plan_tiles(cut,1800,1280)
        self.assertFalse(tiles[0]['usable']);self.assertEqual(tiles[0]['excluded_cut_or_edge_ports'],[0])
        self.assertTrue(tiles[1]['usable']);self.assertEqual(len(tiles[1]['labels']),1)

    def test_empty_port_labels_are_not_erased_other_targets(self):
        boxes=parse_boxes('1 .5 .5 .2 .2',1280,1280)
        tiles=plan_tiles(boxes,1280,1280)
        self.assertTrue(tiles[0]['usable']);self.assertEqual(tiles[0]['labels'],[])
        self.assertEqual(boxes[0]['source_class'],1)

    def test_artificial_edge_guard_but_true_boundary_allowed(self):
        boxes=parse_boxes('4 .3 .5 .022222222222222222 .1',1800,1280)
        tiles=plan_tiles(boxes,1800,1280)
        self.assertTrue(tiles[0]['usable']);self.assertFalse(tiles[1]['usable'])


if __name__=='__main__':unittest.main()
