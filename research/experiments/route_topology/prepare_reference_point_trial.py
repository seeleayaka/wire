"""Pin one interactive decoder trial, reviewed reference only, no prompt sweep."""
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageDraw
from core import sha256,image_binding
from run_review import read,save,verified_run
from run_prompt_contrast import verify
from bundle_runtime_pins import source_pins
from reference_point_seeds import select_points

ROOT=Path(__file__).resolve().parents[2]


def main():
    old=ROOT/'artifacts/mendeley_harness_reference_trial_20261006'
    p=read(old/'protocol.json');verify(p['pins'])
    if source_pins()!=p['mainline_pins']:raise ValueError('E drift')
    run=old/'wire_harness_plus_reference_anatomy_box'
    native=verified_run(run,p['source']['path'])
    readiness=read(ROOT/'artifacts/mendeley_harness_reference_replay_20261006/report.json')
    verify(readiness['pins'])
    if not readiness['reference_boxed_feasibility_passed'] or len(native['paths'])!=1:
        raise ValueError('unique verified reference object required for prompt derivation')
    scope=read(p['confirmed_scope_path'])
    with Image.open(native['paths'][0]) as im:raw=np.asarray(im.convert('L'))
    points,negatives=select_points(raw,scope['anchors'],p['crop_box_xyxy'][:2])
    output=ROOT/'artifacts/mendeley_reference_point_trial_20261006';output.mkdir(exist_ok=False)
    with Image.open(p['original_source']['path']) as im:crop=im.convert('RGB').crop(p['crop_box_xyxy'])
    crop.save(output/'fresh_original_crop.png')
    view=crop.copy();draw=ImageDraw.Draw(view)
    for point in points:
        x,y=point['xy_crop'];draw.ellipse((x-4,y-4,x+4,y+4),fill='#23bc6f',outline='white')
    for x,y in negatives:draw.ellipse((x-4,y-4,x+4,y+4),fill='#d45757',outline='white')
    view.save(output/'reference_prompt_preview.png')
    files=[Path(__file__),Path(__file__).with_name('reference_point_seeds.py'),
           Path(__file__).with_name('run_reference_point_trial.py'),
           Path(__file__).with_name('audit_reference_point_trial.py'),
           native['paths'][0],p['original_source']['path'],p['confirmed_scope_path'],
           output/'fresh_original_crop.png']
    files += [Path(p['sam_source'])/'sam3'/f for f in ['model_builder.py','model/sam3_image.py','model/sam1_task_predictor.py','model/sam3_image_processor.py']]
    pins={**p['pins'],**{str(f):sha256(f) for f in files}}
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(), 'kind':'reference_only_interactive_feasibility',
        'source':{'path':str(output/'fresh_original_crop.png'),**image_binding(output/'fresh_original_crop.png')},
        'original_source':p['original_source'],'crop_box_xyxy':p['crop_box_xyxy'],
        'checkpoint':p['checkpoint'],'sam_source':p['sam_source'],'confirmed_scope_path':p['confirmed_scope_path'],
        'positive_seeds':points,'negative_seeds_crop':negatives,
        'point_coords_crop':[r['xy_crop'] for r in points]+negatives,'point_labels':[1]*len(points)+[0]*len(negatives),
        'box_xyxy_crop':[40,30,350,360],'multimask_output':False,'fresh_encoder_required':True,
        'sam_state_or_embedding_reused':False,'reference_mask_only_used_to_select_prompt_locations':True,
        'points_are_prompts_not_connection_evidence':True,'model_observer_count':1,
        'score_type':'predicted_mask_iou_not_semantic_probability','quality_readiness_min':.75,
        'all_native_outputs_retained':True,'no_mask_bridging_or_geometry_relaxation':True,
        'stop_if':'reference native output has no unique non-boundary two-anchor component; no prompt sweep',
        'source_controls_and_score_calibration_required_before_adoption':True,
        'electric_correctness':'not_assessed','deployed':False,'pins':pins,'mainline_pins':p['mainline_pins'],
        'official_API_reference':'https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam1_task_predictor.py'}
    save(output/'protocol.json',protocol)
    print({'protocol':str(output/'protocol.json'),'positive_reference_points':points})


if __name__=='__main__':main()
