"""Seeded software invariance checks; NOT additional cabinet data/accuracy."""
import unittest
import cv2
import numpy as np

from core import extract_mask,assess_view,compare_views
from test_core import REVIEW
from run_identity_tiles import fixed_regions


class GeneralRoutes(unittest.TestCase):
    def test_seeded_variable_frame_position_identity_and_routing(self):
        rng=np.random.default_rng(20261005)
        for case in range(256):
            width,height=map(int,rng.integers(120,241,size=2))
            ax,bx=20,width-20;y=int(rng.integers(15,25));cy=height-18
            detour=int(rng.integers(y+12,height-40))
            points={'A':(ax,y),'B':(bx,y),'C':(bx,cy)}
            identities={key:f'fixture-{case}:{key}' for key in points}
            images=[]
            for vertices in [[points['A'],points['B']],
                             [points['A'],(ax,detour),(bx,detour),points['B']],
                             [points['A'],(ax,cy),points['C']]]:
                array=np.zeros((height,width),np.uint8)
                cv2.polylines(array,[np.asarray(vertices,np.int32)],False,255,1)
                images.append(array)
            interrupted=images[1].copy();interrupted[:,(ax+bx)//2-2:(ax+bx)//2+3]=0
            images.append(interrupted)
            entries=[{'id':identities[key],'roi_kind':'wire_entry_port',
                      'bbox_xyxy':[x-3,y0-3,x+3,y0+3],**REVIEW} for key,(x,y0) in points.items()]
            # Apply the same randomly chosen exact rigid pixel transform to
            # every input, including independently bound port rectangles.
            rotation=int(rng.integers(0,4))
            def rotate(x,y0):
                return [(x,y0),(y0,width-1-x),(width-1-x,height-1-y0),(height-1-y0,x)][rotation]
            for entry in entries:
                l,t,r,b=entry['bbox_xyxy'];corners=[rotate(x,y0) for x,y0 in [(l,t),(l,b),(r,t),(r,b)]]
                entry['bbox_xyxy']=[min(p[0] for p in corners),min(p[1] for p in corners),
                                   max(p[0] for p in corners),max(p[1] for p in corners)]
            images=[np.rot90(image,rotation).copy() for image in images]
            size=[images[0].shape[1],images[0].shape[0]]
            views=[assess_view([extract_mask(image,.9,f'case-{case}-mask-{index}')],entries,size,REVIEW)
                   for index,image in enumerate(images)]
            expected=[{'from':identities['A'],'to':identities['B']}]
            for index,decision in [(1,'same_visible_terminal_relations'),
                                   (2,'visible_terminal_relation_difference'),(3,'insufficient_evidence')]:
                with self.subTest(case=case,variant=index,rotation=rotation,size=size):
                    result=compare_views(views[0],views[index],expected,REVIEW)
                    self.assertEqual(result['decision'],decision)
                    self.assertFalse(result['automatic_fault_verdict'])
                    self.assertEqual(result['electrical_continuity'],'not_assessed')

    def test_fixed_ocr_regions_cover_odd_and_small_frame_without_scene_rules(self):
        for w,h in [(101,99),(620,622),(575,517),(3648,2736),(3,3)]:
            regions=list(fixed_regions(w,h))
            self.assertEqual(len(regions),10)
            self.assertEqual(regions[0],('full',[0,0,w,h]))
            cover=np.zeros((h,w),bool)
            for _,(l,t,r,b) in regions[1:]:
                self.assertTrue(0<=l<r<=w and 0<=t<b<=h)
                cover[t:b,l:r]=True
            self.assertTrue(cover.all())


if __name__=='__main__':
    unittest.main(verbosity=2)
