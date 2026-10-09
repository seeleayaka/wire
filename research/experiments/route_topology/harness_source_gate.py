"""Strict source readiness for the distinct harness recipe, not an adoption."""
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256
from run_review import read, save, verified_run
from run_prompt_contrast import verify
from bundle_runtime_pins import source_pins
from visible_harness_relation import observe_harness, compare_harness

ROOT = Path(__file__).resolve().parents[2]


def main():
    evidence=ROOT/'artifacts/mendeley_harness_source_controls_20261006'
    p=read(evidence/'protocol.json')
    inference=read(evidence/'inference_report.json')
    if inference['status']!='complete' or inference['protocol_sha256']!=sha256(evidence/'protocol.json'):
        raise ValueError('fresh harness source inference incomplete')
    verify(p['pins'])
    scope=read(p['confirmed_scope_path'])
    observations=[]
    rows=read(evidence/'preparation_report.json')['cases']
    native_count=0
    for row in rows:
        masks=[]
        if row['sam_inference_requested']:
            for recipe in p['recipes']:
                run=evidence/row['id']/recipe
                manifest=read(run/'run_manifest.json')
                native=verified_run(run,manifest['image_binding']['image_path'])
                native_count+=len(native['paths'])
                if recipe!='wire_harness_plus_reference_anatomy_box':
                    continue
                for index,(path,score) in enumerate(zip(native['paths'],native['scores'])):
                    with Image.open(path) as im:
                        raw=np.asarray(im.convert('L'))
                    masks.append({'record_id':f'mask_{index+1:03d}', 'raw':raw,
                                  'score':score,'recipe':recipe})
        binding={k:row['original_source'][k] for k in ['image_sha256','image_size','coordinate_frame']}
        observations.append(observe_harness(scope,binding,row['anchors'],masks,row['phenotype'],
            translation=tuple(row['crop_context']['crop_box_xyxy'][:2]),
            sam_inventory_verified=row['sam_inference_requested']))
    results=[]
    for row,observation in zip(rows,observations):
        annotation=row['source_control_annotation']
        expected='same_visible_bundle_attachment_supported' if annotation is None or annotation['visual_label']==1 else 'visible_socket_attachment_change_supported'
        decision=compare_harness(scope,observations[0],observation)
        results.append({'id':row['id'],'source_annotation':annotation,'observation':observation,
                        'comparison':decision,'expected_source_decision':expected,
                        'control_passed':decision['decision']==expected})
    if source_pins()!=p['mainline_pins']:
        raise ValueError('E drift')
    verify(p['pins'])
    output=ROOT/'artifacts/mendeley_harness_source_gate_20261006'
    output.mkdir(exist_ok=False)
    save(output/'report.json',{'status':'complete','source_gate_passed':all(r['control_passed'] for r in results),
        'cases':results,'native_masks_verified':native_count,'new_recipe_not_relabelled_as_cable':True,
        'old_any_high_score_socket_touch_conflict_preserved':True,
        'no_extent_only_override':True,'not_independent_field_accuracy':True,
        'source_annotations_assistant_qualitative':True,'deployed':False,'electrical_correctness':'not_assessed',
        'pins':{str(f):sha256(f) for f in [Path(__file__),evidence/'protocol.json',
                  evidence/'preparation_report.json',evidence/'inference_report.json',
                  Path(__file__).with_name('visible_harness_relation.py')]}})
    print({'source_gate_passed':all(r['control_passed'] for r in results)})


if __name__=='__main__':
    main()
