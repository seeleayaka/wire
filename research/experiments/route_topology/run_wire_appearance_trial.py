"""Freeze appearance from reference only, replay all30; no SAM calls."""
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw
from run_prompt_contrast import save,digest,verify
from run_review import verified_run
from bundle_runtime_pins import source_pins
from reference_wire_appearance import fit_reference,assess,POLICY

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'artifacts/endpoint_pair_expanded30_v3_20261008'
OUT=ROOT/'artifacts/wire_appearance_reference_20261008'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def region_masks(shape,crop,scope,poses):
    ys,xs=np.indices(shape)
    points=np.column_stack((xs.ravel()+crop[0],ys.ravel()+crop[1],np.ones(xs.size)))
    regions={};scales={}
    for anchor in scope['anchors']:
        pose=next(p for p in poses if p['id']==anchor['id'])
        if not pose['localization_proposal_supported'] or not all(pose['gates'].values()):continue
        H=np.array(pose['inspection_to_reference_local'])
        q=points@H.T
        assert (np.abs(q[:,2])>1e-9).all()
        xy=q[:,:2]/q[:,2:]
        l,t,r,b=anchor['bbox_xyxy'];cx=(l+r)/2;cy=(t+b)/2
        # Fixed2x rectangle about the reviewed anchor, not case-dependent tuning.
        inside=(abs(xy[:,0]-cx)<=r-l)&(abs(xy[:,1]-cy)<=b-t)
        regions[anchor['id']]=inside.reshape(shape)
        denominator=np.median(q[inside,2])
        scales[anchor['id']]=float(np.sqrt(abs(np.linalg.det(H)/denominator**3)))
    return regions,scales


def main():
    started=time.monotonic()
    prior=read(INPUT/'report.json');p=read(INPUT/'protocol.json');verify(p['pins'])
    assert prior['status']=='complete' and len(prior['cases'])==30
    before=source_pins();assert before==p['mainline_pins']
    base=ROOT/'artifacts/mendeley_confirmed_bundle_batch_v2_20261006'
    recent=ROOT/'artifacts/mendeley_semantic_visible_bundle_20261006'
    scope=read(ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json')
    prepared={r['id']:r for r in read(base/'preparation_report.json')['cases']}
    prepared.update({r['id']:r for r in read(recent/'preparation_report.json')['cases']})
    ref=prepared['reference'];crop=ref['crop_context']['crop_box_xyxy']
    rgb=np.asarray(Image.open(ref['original_source']['path']).convert('RGB').crop(crop))
    inv=verified_run(base/'reference/cable_plus_reference_anatomy_box',ref['crop_context']['source']['path'])
    ref_masks=[np.asarray(Image.open(path).convert('L'))>0 for path,score in zip(inv['paths'],inv['scores']) if score>=.75]
    assert len(ref_masks)==1
    regions,scales=region_masks(rgb.shape[:2],crop,scope,ref['anchors'])
    profiles={k:fit_reference(rgb,ref_masks[0]&region) for k,region in regions.items()}
    OUT.mkdir(exist_ok=False)
    pins={str(path):digest(path) for path in [Path(__file__),Path(__file__).with_name('reference_wire_appearance.py'),
        INPUT/'report.json',INPUT/'protocol.json',Path(scope['reference_image_path'])]}
    save(OUT/'protocol.json',dict(pins=pins,mainline_pins=before,policy=POLICY,
        reference_profiles=profiles,fitted_to_reference_only=True,source_cases=30,fresh_SAM=False))
    results=[]
    for row in prior['cases']:
        case=prepared[row['id']];endpoint_results={};component_results=[];flagged=[]
        if row['native_run_directory']:
            crop=row['crop_box_xyxy'];source=row['source_path']
            assert digest(source)==row['source_binding']['image_sha256']
            rgb=np.asarray(Image.open(source).convert('RGB').crop(crop))
            context=case['crop_context']
            assert np.array_equal(rgb,np.asarray(Image.open(context['source']['path']).convert('RGB')))
            inv=verified_run(row['native_run_directory'],context['source']['path'])
            regions,scales=region_masks(rgb.shape[:2],crop,scope,case['anchors'])
            candidate_id=row['candidate_records'][0]['record_id'] if row['decision']=='reference_endpoint_pair_candidate' else None
            display=rgb.copy();supported_pixels=np.zeros(rgb.shape[:2],bool);unsupported_pixels=np.zeros_like(supported_pixels)
            for path,score in zip(inv['paths'],inv['scores']):
                if score<.75:continue
                raw=np.asarray(Image.open(path).convert('L'))>0
                if path.stem==candidate_id:
                    for identity,region in regions.items():
                        metrics,match=assess(rgb,raw&region,profiles[identity],scales[identity])
                        endpoint_results[identity]=metrics
                        supported_pixels|=match
                n,labels,stats,_=cv2.connectedComponentsWithStats(raw.astype('uint8'),connectivity=8)
                for c in range(1,n):
                    component=labels==c
                    # Union is a palette union only: original masks/components are NEVER unioned.
                    combined=dict(profiles[next(iter(profiles))])
                    combined['allowed_hue_bins']=sorted(set(b for profile in profiles.values() for b in profile['allowed_hue_bins']))
                    combined['width_p95']=max(profile['width_p95'] for profile in profiles.values())
                    metrics,match=assess(rgb,component,combined)
                    overlap={k:int((component&region).sum()) for k,region in regions.items()}
                    remote=not any(overlap.values())
                    item=dict(record_id=path.stem,component_index=c,pixels=int(stats[c,cv2.CC_STAT_AREA]),
                        expanded_endpoint_overlap=overlap,remote_from_endpoints=remote,appearance=metrics,
                        component_mask_sha256=__import__('hashlib').sha256(component.tobytes()).hexdigest())
                    component_results.append(item)
                    if remote and metrics['state']!='reference_color_supported':
                        flagged.append(dict(record_id=path.stem,component_index=c,state=metrics['state'],pixels=item['pixels']))
                        unsupported_pixels|=component
            display[supported_pixels]=(display[supported_pixels]*.4+np.array([0,220,100])*.6).astype('uint8')
            display[unsupported_pixels]=(display[unsupported_pixels]*.4+np.array([240,0,180])*.6).astype('uint8')
            canvas=Image.new('RGB',(rgb.shape[1],rgb.shape[0]+90),'white')
            canvas.paste(Image.fromarray(display),(0,0));draw=ImageDraw.Draw(canvas)
            draw.text((8,rgb.shape[0]+8),row['id']+' appearance diagnostic only',fill='black')
            draw.text((8,rgb.shape[0]+30),'green: reference colors; purple: remote unsupported',fill='black')
            draw.text((8,rgb.shape[0]+52),'original masks kept; physical identity UNKNOWN',fill='black')
            canvas.save(OUT/(row['id']+'.png'))
        appearance_ok=bool(row['decision']=='reference_endpoint_pair_candidate' and len(endpoint_results)==2
                           and all(r['state']=='reference_color_supported' for r in endpoint_results.values()))
        results.append(dict(id=row['id'],original_endpoint_decision=row['decision'],endpoint_appearance=endpoint_results,
            appearance_supported_candidate=appearance_ok,remote_unsupported_components=flagged,
            all_high_score_components=component_results,same_physical_wire_confirmed=False,
            native_masks_changed=False,electrical_continuity='not_assessed'))
    verify(pins);assert source_pins()==before
    supported=[r['id'] for r in results if r['appearance_supported_candidate']]
    rejected=[r['id'] for r in results if r['original_endpoint_decision']=='reference_endpoint_pair_candidate'
              and not r['appearance_supported_candidate']]
    report=dict(status='complete',cases=results,appearance_supported_candidates=len(supported),
        supported_ids=supported,old_endpoint_candidates=19,appearance_unconfirmed_candidate_ids=rejected,
        remote_component_flag_case_ids=[r['id'] for r in results if r['remote_unsupported_components']],
        original_endpoint_report_unchanged=True,no_deployment=True,physical_identity_confirmed=0,
        fresh_SAM_calls=0,seconds=time.monotonic()-started,actual_visual_review='pending',
        independent_feature_audit='pending',not_field_accuracy=True)
    save(OUT/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))


if __name__=='__main__':main()
