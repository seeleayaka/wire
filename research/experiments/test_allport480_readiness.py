import copy
import unittest
from allport480_readiness import independent_labels, legacy_labels, plan_all


def port(i, box, cls=0):
    return dict(source_box_index=i, class_id=cls, box_xyxy=box)


class ReadinessTests(unittest.TestCase):
    def test_legacy_retains_clipped_label_new_policy_refuses(self):
        ports=[port(7,[150,90,200,130]),port(8,[300,300,330,340],1)]
        window=[100,100,580,580]
        old=legacy_labels(ports,window)
        self.assertEqual(len(old),2)
        self.assertFalse(old[0]['complete'])
        self.assertEqual(old[0]['box_xyxy_local'],[50,0,100,30])
        self.assertTrue(old[1]['complete'])
        self.assertIsNone(independent_labels(ports,window,[1000,1000]))

    def test_partial_visible_is_not_erased(self):
        self.assertIsNone(independent_labels([port(0,[50,50,200,200])], [100,100,580,580], [1000,1000]))

    def test_artificial_guard_inclusive(self):
        self.assertIsNone(independent_labels([port(0,[116,150,180,190])], [100,100,580,580], [1000,1000]))
        self.assertEqual(len(independent_labels([port(0,[117,150,180,190])], [100,100,580,580], [1000,1000])),1)

    def test_physical_edge_allowed(self):
        self.assertEqual(len(independent_labels([port(0,[0,0,20,20])], [0,0,480,480], [480,480])),1)

    def test_small_source_no_padding(self):
        self.assertEqual(plan_all([port(0,[0,0,20,20])],[300,600]),([], [0]))

    def test_order_and_inputs(self):
        ports = [port(7,[300,300,330,350]), port(3,[400,310,430,340],1)]
        before = copy.deepcopy(ports)
        a = plan_all(ports,[1200,1200])
        self.assertEqual(a,plan_all(list(reversed(ports)),[1200,1200]))
        self.assertEqual(ports,before)
        self.assertEqual({p['source_box_index'] for r in a[0] for p in r['labels']},{3,7})
        self.assertEqual(len({tuple(r['window']) for r in a[0]}),len(a[0]))

    def test_translation_and_class_preserved(self):
        a = independent_labels([port(0,[150,160,200,210],1)],[100,100,580,580],[1000,1000])
        b = independent_labels([port(0,[250,260,300,310],1)],[200,200,680,680],[1100,1100])
        self.assertEqual(a,b)
        self.assertEqual(a[0]['class_id'],1)

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            independent_labels([], [0,0,480,480],[300,300])


if __name__ == '__main__':
    unittest.main()
