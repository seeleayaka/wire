"""SAM-shaped reports must reach the port selector without changing baseline evidence."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from inspection_agent.port_crop_gui_bridge import run_gui_port_review, render_port_overlay
from inspection_agent.port_tiling import select_additional_tile_hint


def report(parent=None):
    return {'review_regions': [parent or {'id':'green_01', 'bbox_xyxy':[10,20,110,120],
                'tier':'dino_and_sam', 'not_claimed':['electrical_continuity']}],
            'existing_port_hints':[], 'alignment':{'source_to_reference_homography':np.eye(3).tolist()},
            'alignment_quality':{'reliable':True}, 'local_alignment':[],
            'image_fingerprints':{'stable_during_visual_analysis':True,
                'source_sha256':'source', 'reference_sha256':'reference'}}


def selector(source, reference, context, **kwargs):
    prediction=dict(left=30,top=40,right=40,bottom=50,confidence=.8,class_id=0,valid_warp_fraction=1.)
    result=select_additional_tile_hint(context['parents'],context['existing_hints'],[prediction],.25)
    return {**result,'status':'applied','fallback_reason':None,'automatic_fault_verdict':False}


class PortGuiBridgeTests(unittest.TestCase):
    def invoke(self, value, enabled=True):
        with patch('inspection_agent.port_crop_gui_bridge.optional_port_crop_review',side_effect=selector) as call:
            result=run_gui_port_review(value,project=Path('.'),enabled=enabled,scene='test')
        return result,call

    def test_sam_shape_selected_with_original_parents_preserved(self):
        value=report(); before=copy.deepcopy(value)
        result,call=self.invoke(value)
        self.assertEqual(result['status'],'applied')
        self.assertEqual(len(result['tile_hints']),1)
        self.assertEqual(result['tile_hints'][0]['parent_index'],0)
        self.assertEqual(result['parents'],before['review_regions'])
        self.assertEqual(value,before)
        self.assertEqual(call.call_args.args[2]['parents'][0]['tier'],'dino_and_sam')

    def test_legacy_shape_preserved(self):
        value=report(dict(left=10,top=20,right=110,bottom=120,id='old'))
        result,_=self.invoke(value)
        self.assertEqual(result['parents'],value['review_regions'])
        self.assertEqual(len(result['tile_hints']),1)

    def test_consistent_dual_shape_and_empty_candidates(self):
        parent=dict(bbox_xyxy=[10,20,110,120],left=10,top=20,right=110,bottom=120)
        result,_=self.invoke(report(parent))
        self.assertEqual(result['parents'],[parent])
        value=report();value['review_regions']=[]
        result,_=self.invoke(value)
        self.assertEqual(result['parents'],[]);self.assertEqual(result['tile_hints'],[])

    def test_missing_provenance_does_not_call_model(self):
        value=report();value['image_fingerprints']={}
        result,call=self.invoke(value)
        self.assertEqual(result['fallback_reason'],'visual_geometry_provenance_missing')
        self.assertEqual(result['parents'],value['review_regions']);call.assert_not_called()

    def test_downstream_fallback_restores_original_evidence(self):
        value=report();before=copy.deepcopy(value)
        def failed(*args,**kwargs):
            return dict(status='fallback',parents=copy.deepcopy(args[2]['parents']),
                existing_hints=[],tile_hints=[],fallback_reason='unsupported_scene')
        with patch('inspection_agent.port_crop_gui_bridge.optional_port_crop_review',side_effect=failed):
            result=run_gui_port_review(value,project=Path('.'),enabled=True)
        self.assertEqual(result['parents'],before['review_regions']);self.assertEqual(value,before)

    def test_disabled_and_local_ecc_do_not_call_model(self):
        value=report()
        result,call=self.invoke(value,False)
        self.assertEqual(result['status'],'disabled');call.assert_not_called()
        value['local_alignment']=[{'dino_alignment_input':'local_ecc_corrected'}]
        result,call=self.invoke(value)
        self.assertEqual(result['fallback_reason'],'local_alignment_not_supported');call.assert_not_called()

    def test_malformed_or_ambiguous_geometry_rejected_before_model(self):
        for coords in ([1,2,1,3],[-1,2,3,4],[1,2,float('nan'),4],
                       [True,2,3,4],[1,2,3],['1',2,3,4]):
            with self.subTest(coords=coords):
                result,call=self.invoke(report({'bbox_xyxy':coords}))
                self.assertEqual(result['status'],'fallback');call.assert_not_called()
                self.assertEqual(result['fallback_reason'],'invalid_parent_geometry')
        result,call=self.invoke(report(dict(bbox_xyxy=[10,20,110,120],left=11,top=20,right=110,bottom=120)))
        self.assertEqual(result['fallback_reason'],'invalid_parent_geometry');call.assert_not_called()
        result,call=self.invoke(report(dict(bbox_xyxy=[1,2,3,4],left=True,top=2,right=3,bottom=4)))
        self.assertEqual(result['fallback_reason'],'invalid_parent_geometry');call.assert_not_called()

    def test_overlay_supports_original_sam_parents_without_mutation(self):
        value={'parents':report()['review_regions'],'tile_hints':[]}
        before=copy.deepcopy(value)
        with tempfile.TemporaryDirectory() as folder:
            with patch('inspection_agent.port_crop_gui_bridge.read_image',return_value=np.zeros((140,140,3),np.uint8)):
                output=render_port_overlay('unused.jpg',value,Path(folder)/'overlay.jpg')
            self.assertTrue(output.is_file())
        self.assertEqual(value,before)


if __name__=='__main__':unittest.main()
