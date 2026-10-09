"""Recompute pixel histograms and width statistics; HSV/distance primitives shared."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from run_prompt_contrast import digest,save,verify
from run_review import verified_run
from bundle_runtime_pins import source_pins
from run_wire_appearance_trial import region_masks
from audit_semantic_visible_bundle_native import flood_components

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/wire_appearance_reference_v3_20261008'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def statistics(rgb,mask,profile,scale):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    hist=[0]*18
    for h,s,v in hsv[mask].tolist():
        if s>=64 and v>=32:hist[h//10]+=1
    count=sum(hist)
    fraction=sum(hist[b] for b in profile['allowed_hue_bins'])/count if count else 0.
    density=count/int(mask.sum()) if mask.any() else 0.
    widths=(2*cv2.distanceTransform(np.pad(mask.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1][mask]*scale).tolist()
    widths.sort()
    n=len(widths)
    median=(widths[n//2] if n%2 else (widths[n//2-1]+widths[n//2])/2) if n else None
    wide=median is not None and median>profile['width_p95']*2
    state=('reference_color_supported' if count>=16 and fraction>=.5 and not wide else
           'insufficient_colored_evidence' if count<16 else 'appearance_mismatch_or_overwide')
    if state=='reference_color_supported' and density<profile['minimum_colored_fraction']:
        state='insufficient_colored_coverage'
    return state,count,fraction,density,median


def check(rgb,mask,profile,scale,old):
    state,count,fraction,density,median=statistics(rgb,mask,profile,scale)
    assert state==old['state'] and count==old['colored_pixels']
    assert abs(fraction-old['matched_color_fraction'])<1e-10
    assert abs(density-old['colored_fraction'])<1e-10
    assert (median is None and old['median_width_reference_pixels'] is None) or abs(median-old['median_width_reference_pixels'])<1e-4


def main():
    p=read(OUT/'protocol.json');verify(p['pins'])
    report=read(OUT/'report.json');source=read(ROOT/'artifacts/endpoint_pair_expanded30_v3_20261008/report.json')
    base=ROOT/'artifacts/mendeley_confirmed_bundle_batch_v2_20261006';recent=ROOT/'artifacts/mendeley_semantic_visible_bundle_20261006'
    prepared={r['id']:r for r in read(base/'preparation_report.json')['cases']}
    prepared.update({r['id']:r for r in read(recent/'preparation_report.json')['cases']})
    scope=read(ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json')
    profiles=p['reference_profiles'];endpoints=0;components=0
    for row in report['cases']:
        prior=next(r for r in source['cases'] if r['id']==row['id'])
        if not prior['native_run_directory']:continue
        case=prepared[row['id']];crop=prior['crop_box_xyxy']
        assert digest(prior['source_path'])==prior['source_binding']['image_sha256']
        rgb=np.asarray(Image.open(prior['source_path']).convert('RGB').crop(crop))
        regions,scales=region_masks(rgb.shape[:2],crop,scope,case['anchors'])
        inv=verified_run(prior['native_run_directory'],case['crop_context']['source']['path'])
        for path,score in zip(inv['paths'],inv['scores']):
            if score<.75:continue
            raw=np.asarray(Image.open(path).convert('L'))>0
            if prior['candidate_records'] and path.stem==prior['candidate_records'][0]['record_id']:
                for anchor,old in row['endpoint_appearance'].items():
                    check(rgb,raw&regions[anchor],profiles[anchor],scales[anchor],old);endpoints+=1
            combined=dict(profiles[next(iter(profiles))])
            combined['allowed_hue_bins']=sorted(set(b for profile in profiles.values() for b in profile['allowed_hue_bins']))
            combined['width_p95']=max(profile['width_p95'] for profile in profiles.values())
            for part in flood_components(raw):
                mask=np.zeros(raw.shape,bool);mask.ravel()[part]=True
                sha=__import__('hashlib').sha256(mask.tobytes()).hexdigest()
                old=next(c for c in row['all_high_score_components'] if c['record_id']==path.stem and c['component_mask_sha256']==sha)
                check(rgb,mask,combined,1.,old['appearance']);components+=1
        assert row['same_physical_wire_confirmed'] is False and row['native_masks_changed'] is False
    verify(p['pins']);assert source_pins()==p['mainline_pins']
    save(OUT/'audit_report.json',dict(status='PASS',cases=30,endpoint_descriptors=endpoints,component_descriptors=components,
        scalar_histograms_and_sorted_width_statistics_recomputed=True,components_independent_flood_fill=True,
        shared_primitives=['OpenCV HSV','OpenCV distance transform','registered region mapping'],
        reference_palette_fitting_not_independently_reimplemented=True,
        report_sha256=digest(OUT/'report.json'),no_field_accuracy_claim=True,fresh_SAM=False))
    print('PASS',endpoints,'endpoint descriptors;',components,'native component descriptors')


if __name__=='__main__':main()
