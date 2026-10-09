"""Original RGB review packet, no mask/prediction overlays or GT rewriting."""
import json
from pathlib import Path
from PIL import Image,ImageDraw
from run_prompt_contrast import save,digest,verify
from bundle_runtime_pins import source_pins

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/wire_identity_photo_review_20261008'
INPUT=ROOT/'artifacts/endpoint_pair_expanded30_v3_20261008/report.json'
SCOPE=ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json'


def main():
    before=source_pins();prior=json.loads(INPUT.read_text(encoding='utf-8'))
    scope=json.loads(SCOPE.read_text(encoding='utf-8'))
    assert prior['status']=='complete' and len(prior['cases'])==30
    OUT.mkdir(exist_ok=False)
    pins={str(p):digest(p) for p in [Path(__file__),INPUT,SCOPE]}
    samples=[]
    for i,row in enumerate(prior['cases'],1):
        source=Path(row['source_path']);sha=digest(source)
        assert sha==row['source_binding']['image_sha256']
        original=Image.open(source).convert('RGB');w,h=original.size
        crop=row['crop_box_xyxy']
        provenance='existing_ROI_only_no_prediction_overlay'
        if crop is None:
            points=[p for quad in row['endpoint_source_quads'].values() for p in quad]
            if points:
                xs=[p[0] for p in points];ys=[p[1] for p in points]
                crop=[max(0,int(min(xs)-200)),max(0,int(min(ys)-200)),min(w,int(max(xs)+200)),min(h,int(max(ys)+200))]
                provenance='existing_endpoint_proposal_context_not_independent_crop'
            else:crop=[1300,850,min(w,1900),min(h,1450)];provenance='reference_coordinate_context_fallback'
        identity=f'P{i:02d}'
        original.crop(crop).save(OUT/(identity+'_detail.png'))
        overview=original.copy();overview.thumbnail((1000,750));overview.save(OUT/(identity+'_overview.png'))
        samples.append(dict(review_id=identity,case_id=row['id'],source_path=str(source),source_sha256=sha,
                            crop_box_xyxy=crop,crop_origin=provenance,
                            detail_image=identity+'_detail.png',overview_image=identity+'_overview.png'))
    ref=Image.open(scope['reference_image_path']).convert('RGB')
    ref.crop((1400,920,1800,1320)).save(OUT/'reference_detail.png')
    for start in range(0,30,5):
        canvas=Image.new('RGB',(2050,455),'white');draw=ImageDraw.Draw(canvas)
        for i,sample in enumerate(samples[start:start+5]):
            image=Image.open(OUT/sample['detail_image']).convert('RGB');image.thumbnail((400,400))
            canvas.paste(image,(410*i,0));draw.text((410*i+8,410),sample['review_id']+' original RGB, no masks',fill='black')
        canvas.save(OUT/f'sheet_{start//5+1:02d}.png')
    manifest=dict(status='prepared',samples=samples,reference_binding=scope['reference_binding'],
        packet_has_no_prediction_or_mask_overlay=True,reviewer_has_prior_project_context=True,
        not_reviewer_blinded=True,not_independent_physical_GT=True,mainline_pins=before,pins=pins,
        fresh_SAM_calls=0,original_sources_unchanged=True)
    verify(pins);assert source_pins()==before
    save(OUT/'manifest.json',manifest)
    print('Prepared30 original photo contexts +30 overviews + reference; no SAM or model overlay.')


if __name__=='__main__':main()
