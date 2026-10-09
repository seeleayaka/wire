"""Inspect one fixed gate failure's actual observed/expected crop pixels.

Diagnostic only: no model, fitting, threshold/image-name exception or GT edit.
It is a dataset normal-control cue, NOT a measured physical electrical fault.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,DATA,REPO,load,save,sha,read_image,REFERENCE_SHA
from inspection_agent.paired_port_features import expected_in_source,crop_tensor,CONTEXT_SCALES
from PIL import Image,ImageDraw,ImageFont
import numpy as np

SOURCE=ROOT/'artifacts/allport480_source_20261005'
OUT=ROOT/'artifacts/allport480_normal_gate_failure_review_20261005'


def original_model_crop(image,box,scale):
    tensor=crop_tensor(image,box,scale)
    mean=np.array([.485,.456,.406])[:,None,None];std=np.array([.229,.224,.225])[:,None,None]
    rgb=np.clip(np.rint((tensor.numpy()*std+mean)*255),0,255).astype(np.uint8).transpose(1,2,0)
    return Image.fromarray(rgb)


def main():
    if OUT.exists():raise FileExistsError('preserve actual normal-control review')
    path=SOURCE/'train/normal_008_predictions.json';digest=sha(path);case=load(path)
    old=case['current']['all_predictions'];new=case['trial']['all_predictions']
    if old or len(new)!=1:raise ValueError('fixed failure case changed')
    cue=new[0];source=DATA/'images/train01'/case['image'];reference=DATA/'images/train01/normal_073.JPG'
    pins={str(p):sha(p) for p in (Path(__file__),path,source,reference,REPO/'inspection_agent/paired_port_features.py')}
    if pins[str(source)]!=case['source_sha256'] or pins[str(reference)]!=REFERENCE_SHA:raise ValueError('actual pixel identity mismatch')
    observed=read_image(source);ref=read_image(reference)
    expected,_=expected_in_source(ref,case['alignment']['source_to_reference_homography'],observed.shape[:2])
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    sheet=Image.new('RGB',(980,1090),'white');draw=ImageDraw.Draw(sheet)
    draw.text((15,12),'Dataset normal control: 1 NEW unmatched review cue; NOT electrical fault proof',font=font,fill='black')
    draw.text((15,42),'Actual classifier RGB inputs; no image repair or new inference. 224px displayed 2x.',font=small,fill='black')
    draw.text((15,67),f"p={cue['paired_semantic_probability']:.6f}; 3 SHA roles; source gate normal0 FAILS",font=small,fill='black')
    for index,scale in enumerate(CONTEXT_SCALES):
        top=125+index*480
        for column,(name,image) in enumerate([('Observed source',observed),('Warped expected reference',expected)]):
            left=15+column*480
            draw.text((left,top-26),f'{name}, context scale {scale}',font=font,fill='black')
            crop=original_model_crop(image,cue['box_xyxy'],scale)
            sheet.paste(crop.resize((448,448),Image.Resampling.NEAREST),(left,top))
    if any(sha(p)!=value for p,value in pins.items()):raise ValueError('review inputs changed')
    OUT.mkdir();sheet.save(OUT/'actual_classifier_crops.png')
    save(OUT/'report.json',dict(status='complete',source_case_sha256=digest,pins=pins,
        source_image_sha256=case['source_sha256'],normal_control_image=case['image'],cue=cue,
        contexts=list(CONTEXT_SCALES),model_inference=False,new_physical_fault_confirmation=False,
        no_GT_edit=True,no_threshold_change=True,no_deployment=True,field_accuracy=None,
        source_run_still_running=(load(SOURCE/'progress.json')['status']=='running')))
    print(dict(status='complete',output=str(OUT),model_inference=False))


if __name__=='__main__':main()
