import copy
import sys
import unittest
sys.path.insert(0,'E:/PythonProject10')
import torch
from dense_pixel_geometric_augmentation import transform_item


class AugmentationTests(unittest.TestCase):
    def item(self):
        return dict(features={'semantic':torch.arange(16).reshape(1,4,4),'rgb':torch.arange(16).reshape(1,4,4)},
                    shape=[480,640],boxes=[dict(class_id=0,box=[80.,120.,160.,240.])])
    def test_all_eight_transforms_are_invertible_and_no_mutation(self):
        original=self.item();saved=copy.deepcopy(original)
        for r in range(4):
            for m in (False,True):
                changed=transform_item(original,r,m)
                if m:changed=transform_item(changed,0,True)
                restored=transform_item(changed,(-r)%4,False)
                self.assertTrue(torch.equal(restored['features']['rgb'],original['features']['rgb']))
                self.assertEqual(restored['shape'],original['shape'])
                for a,b in zip(restored['boxes'][0]['box'],original['boxes'][0]['box']):self.assertAlmostEqual(a,b)
                self.assertEqual(int(changed['targets']['mask'].sum()),1)
        self.assertEqual(original['boxes'],saved['boxes'])
        self.assertTrue(torch.equal(original['features']['rgb'],saved['features']['rgb']))
    def test_native_coordinate_rotation_and_mirror(self):
        r=transform_item(self.item(),1,False)
        self.assertEqual(r['shape'],[640,480]);self.assertEqual(r['boxes'][0]['box'],[120.,480.,240.,560.])
        r=transform_item(self.item(),0,True)
        self.assertEqual(r['boxes'][0]['box'],[480.,120.,560.,240.])
    def test_empty_and_invalid(self):
        item=self.item();item['boxes']=[]
        self.assertFalse(transform_item(item,3,True)['targets']['mask'].any())
        with self.assertRaises(ValueError):transform_item(item,4,False)


if __name__=='__main__':unittest.main()
