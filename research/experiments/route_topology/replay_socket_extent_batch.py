"""Posthoc frozen native-mask replay, not fresh SAM or blind/field accuracy."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
from bundle_runtime_pins import source_pins
from core import sha256
from run_prompt_contrast import verify
from run_review import read, save, verified_run
from socket_native_extent import POLICY, observe_with_extent, compare_with_extent

ROOT=Path(__file__).resolve().parents[2]


def main():
    evidence=ROOT/'artifacts/mendeley_confirmed_bundle_batch_v2_20261006'
    old_path=ROOT/'artifacts/mendeley_confirmed_bundle_batch_v2_review_20261006/report.json'
    audit_path=ROOT/'artifacts/mendeley_confirmed_bundle_batch_v2_audit_20261006/report.json'
    protocol=read(evidence/'protocol.json'); preparation=read(evidence/'preparation_report.json')
    old=read(old_path); audit=read(audit_path); scope=read(protocol['confirmed_scope_path'])
    verify(protocol['pins']); verify(old['pins']); verify(audit['pins'])
    if audit['status']!='PASS' or source_pins()!=protocol['mainline_pins']:raise ValueError('completed verified baseline/E snapshot required')
    output=ROOT/'artifacts/mendeley_socket_extent_candidate_20261006'; output.mkdir(exist_ok=False)
    files=[Path(__file__),Path(__file__).with_name('socket_native_extent.py'),old_path,audit_path,evidence/'protocol.json',
           evidence/'preparation_report.json',Path(protocol['confirmed_scope_path'])]
    pins={str(p):sha256(p) for p in files}
    save(output/'preregistration.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'policy':POLICY,'pins':pins,
        'source_selection':'all31 fresh-baseline prepared originals, no exclusions','new_SAM_inference':False,
        'posthoc_development_hypothesis_after_case_review':True,'normal_and_unknown_rules_unchanged':True,
        'stop_if':'any old definite decision changes, mask mutation, source/E/pin drift or replay error',
        'adoption_status':'research_candidate_only; no independent source-group validation or E deployment'})
    views=[]; native_count=0
    for row in preparation['cases']:
        original=row['original_source']; case=row.get('crop_context')
        binding={k:original[k] for k in ['image_sha256','image_size','coordinate_frame']}
        if sha256(original['path'])!=binding['image_sha256']:raise ValueError('original photo drift')
        masks=[]
        if row['sam_inference_requested']:
            with Image.open(original['path']) as im:crop=np.asarray(im.convert('RGB').crop(case['crop_box_xyxy']))
            with Image.open(case['source']['path']) as im:stored=np.asarray(im.convert('RGB'))
            if not np.array_equal(crop,stored):raise ValueError('crop differs from original pixels')
            for recipe in protocol['recipes']:
                run=evidence/row['id']/recipe
                manifest=read(run/'run_manifest.json'); verified=verified_run(run,manifest['image_binding']['image_path'])
                native_count+=len(verified['paths'])
                if recipe!='cable_plus_reference_anatomy_box':continue
                for i,path in enumerate(verified['paths']):
                    with Image.open(path) as im:raw=np.asarray(im.convert('L'))
                    masks.append({'raw':raw,'record_id':f'mask_{i+1:03d}','score':verified['scores'][i],'recipe':recipe})
        view=observe_with_extent(scope,binding,row['anchors'],masks,row['phenotype'],
              tuple(case['crop_box_xyxy'][:2]) if case else (0,0),row['sam_inference_requested'])
        views.append({'id':row['id'],'observation':view})
    reference=views[0]['observation']; original_results={r['id']:r for r in old['cases']}
    results=[]; changes=[]
    for item in views[1:]:
        decision=compare_with_extent(scope,reference,item['observation'])
        baseline=original_results[item['id']]['comparison']
        if baseline['decision']!='insufficient_evidence' and baseline['decision']!=decision['decision']:
            raise ValueError('old supported result lost or changed')
        # The old full raw-mask/component audit must be identical, not cleaned.
        if item['observation']['mask_audit']!=original_results[item['id']]['observation']['mask_audit']:
            raise ValueError('native masks/components or their old gates changed')
        results.append({'id':item['id'],'comparison':decision,'baseline_comparison':baseline,'observation':item['observation']})
        if baseline['decision']!=decision['decision']:changes.append(item['id'])
    if native_count!=old['SAM_output_instances']:raise ValueError('raw native inventory differs')
    verify(pins)
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E drift at replay completion')
    counts=dict(Counter(r['comparison']['decision'] for r in results))
    report={'status':'complete','cases':results,'reference_observation':reference,'decision_counts':counts,
        'changed_cases':changes,'old_definite_decisions_preserved':True,'raw_masks_verified':native_count,
        'native_mask_audits_unchanged':True,'fresh_SAM_encoders':0,'baseline_SAM_reused_explicitly':True,
        'posthoc_development_hypothesis_not_blind_validation':True,'E_deployed':False,
        'independent_replay_status':'pending','source_group_gate':'not_run',
        'electrical_correctness':'not_assessed','pins':pins}
    save(output/'report.json',report)
    print({'counts':counts,'changes':changes,'old_results_preserved':True,'new_SAM':False})


if __name__=='__main__':main()
