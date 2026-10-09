"""Independent native pixel geometry of reference prompts; no connection votes."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from run_review import read,save
from run_prompt_contrast import verify
from core import sha256
from replay_harness_reference_trial import parts
from bundle_runtime_pins import source_pins

ROOT=Path(__file__).resolve().parents[2]


def main():
    folder=ROOT/'artifacts/mendeley_reference_point_trial_v2_20261006'
    p=read(folder/'protocol.json');inference=read(folder/'inference_report.json')
    if inference['status']!='complete' or inference['protocol_sha256']!=sha256(folder/'protocol.json'):
        raise ValueError('interactive inference incomplete or protocol changed')
    verify(p['pins'])
    scope=read(p['confirmed_scope_path']);eligible=[];rows=[]
    output=ROOT/'artifacts/mendeley_reference_point_audit_v3_20261006';output.mkdir(exist_ok=False)
    for i,record in enumerate(inference['native_outputs'],1):
        if sha256(record['path'])!=record['sha256']:raise ValueError('native mask changed')
        with Image.open(record['path']) as im:active=np.asarray(im.convert('L'))>0
        height,width=active.shape
        if [width,height]!=p['source']['image_size'] or int(active.sum())!=record['pixels']:
            raise ValueError('native pixel/frame count differs')
        boundary=any(y<=1 or x<=1 or y>=height-2 or x>=width-2 for y,x in zip(*np.nonzero(active)))
        components=[]
        for j,component in enumerate(parts(active),1):
            hits={}
            for anchor in scope['anchors']:
                l,t,r,b=anchor['bbox_xyxy']
                hits[anchor['id']]=int(sum(l<=x+p['crop_box_xyxy'][0]<=r and t<=y+p['crop_box_xyxy'][1]<=b for y,x in component))
            good=bool(not boundary and record['quality_prediction']>=.75 and all(hits.values()))
            if good:eligible.append([i,j])
            components.append({'native_pixels':len(component),'anchor_hits':hits,'reference_geometry_feasible':good})
        rows.append({'native_index':i,'predicted_iou':record['quality_prediction'],'boundary':boundary,'components':components})
        with Image.open(p['source']['path']) as im:rgb=np.asarray(im.convert('RGB')).copy()
        overlay=rgb.astype(float);overlay[active]=overlay[active]*.60+np.array([226,148,60])*.40
        view=Image.fromarray(overlay.astype(np.uint8));draw=ImageDraw.Draw(view)
        for x,y in p['point_coords_crop'][:2]:draw.ellipse((x-3,y-3,x+3,y+3),outline='white')
        view.save(output/f'native_overlay_{i:03d}.png')
    verify(p['pins'])
    if source_pins()!=p['mainline_pins']:raise ValueError('E drift')
    save(output/'report.json',{'status':'complete','reference_geometry_gate_passed':len(eligible)==1,
        'eligible_native_components':eligible,'records':rows,'quality_not_semantic_probability':True,
        'actual_visual_review':'pending','not_source_or_field_accuracy':True,'source_controls_not_run':True,
        'points_are_prompts_not_evidence':True,'deployed':False,'electrical_correctness':'not_assessed',
        'pins':{str(f):sha256(f) for f in [Path(__file__),folder/'protocol.json',folder/'inference_report.json']}})
    print({'reference_geometry_gate_passed':len(eligible)==1})


if __name__=='__main__':main()
