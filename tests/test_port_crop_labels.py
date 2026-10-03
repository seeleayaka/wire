import unittest
from inspection_agent.port_training_audit import parse_boxes
from inspection_agent.port_crop_labels import labeled_tiles,training_tiles


class CropLabelTests(unittest.TestCase):
    def test_cut_labels_retained_and_bounded(self):
        tiles=labeled_tiles(parse_boxes('3 .7 .5 .1 .1',1800,1280),1800,1280)
        self.assertFalse(tiles[0]['labels'][0]['complete'])
        self.assertTrue(tiles[1]['labels'][0]['complete'])
        for t in tiles:
            x,y,r,b=t['window']
            for label in t['labels']:
                l,top,rr,bb=label['box_xyxy_local'];self.assertTrue(0<=l<rr<=r-x and 0<=top<bb<=b-y)

    def test_keeps_all_positive_tiles_and_one_stable_negative(self):
        tiles=labeled_tiles(parse_boxes('4 .1 .1 .02 .02',3648,2736),3648,2736)
        chosen=training_tiles('source.JPG',tiles)
        self.assertEqual(chosen,training_tiles('source.JPG',list(reversed(tiles))))
        self.assertEqual(sum(bool(t['labels']) for t in chosen),sum(bool(t['labels']) for t in tiles))
        self.assertEqual(sum(not t['labels'] for t in chosen),1)

    def test_normal_source_one_tile_and_no_mutation(self):
        tiles=labeled_tiles([],3648,2736)
        self.assertEqual(len(tiles),12);self.assertEqual(len(training_tiles('normal.JPG',tiles)),1)
        self.assertTrue(all(not t['labels'] for t in tiles))

    def test_other_fault_classes_are_not_port_targets(self):
        self.assertFalse(labeled_tiles(parse_boxes('1 .5 .5 .1 .1\n2 .5 .5 .1 .1',1280,1280),1280,1280)[0]['labels'])


if __name__=='__main__':unittest.main()
