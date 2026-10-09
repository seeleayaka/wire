"""Restore each native instance, independently replay pixels, never bridge masks."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256,extract_mask
from quarter_turn_view import forward,restore
from run_prompt_contrast import save,verify
from run_audit import source_pins
from run_paired_evidence import load_run
from visible_bundle_relation import observe_bundle
from audit_semantic_visible_bundle_native import flood_components
from audit_cable_socket_controls import replay_hits
from analyze_mendeley_scope import render_all

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--case',required=True);a=p.parse_args()
    protocol=json.loads(a.protocol.read_text(encoding='utf-8'));out=a.protocol.parent
    verify(protocol['pins']);assert source_pins()==protocol['mainline_pins']
    row=next(r for r in json.loads((out/'preparation_report.json').read_text(encoding='utf-8'))['cases'] if r['id']==a.case)
    context=row['crop_context'];origin=row['original_source'];crop=context['crop_box_xyxy']
    assert sha256(origin['path'])==origin['image_sha256']
    rgb=np.asarray(Image.open(origin['path']).convert('RGB').crop(crop))
    assert np.array_equal(forward(rgb),np.asarray(Image.open(context['source']['path']).convert('RGB')))
    scope=json.loads(Path(protocol['confirmed_scope_path']).read_text(encoding='utf-8'));masks=[];replays=[]
    for recipe in protocol['recipes']:
        run=out/a.case/recipe;_,records=load_run(run);restored=[]
        directory=out/a.case/(recipe+'_restored');directory.mkdir(exist_ok=False)
        for record in records:
            rotated=np.asarray(Image.open(record['source_mask_path']).convert('L'))>0
            raw=restore(rotated,context['original_crop_size'])
            # Independent coordinate formula, not another call to the rotation helper.
            scalar=np.zeros(raw.shape,dtype=bool)
            yy,xx=np.nonzero(rotated);scalar[xx,raw.shape[1]-1-yy]=True
            assert np.array_equal(raw,scalar) and raw.sum()==rotated.sum()
            parts=flood_components(raw)
            assert sorted(map(len,parts))==sorted(map(len,flood_components(rotated)))
            path=directory/(record['record_id']+'.png');Image.fromarray(raw.astype('uint8')*255).save(path)
            r=extract_mask(raw,record['score'],record['record_id']);r['source_mask_path']=str(path);restored.append(r)
            if recipe.endswith('_box'):
                masks.append(dict(raw=raw,record_id=r['record_id'],score=r['score'],recipe=recipe))
                replays.append(dict(record_id=r['record_id'],signatures=replay_hits(parts,raw.shape[1],crop[:2],scope['anchors'],row['anchors'])))
        render_all({'image_path':context['original_crop_path']},restored,directory/'visuals')
    binding={k:origin[k] for k in ['image_sha256','image_size','coordinate_frame']}
    view=observe_bundle(scope,binding,row['anchors'],masks,row['phenotype'],translation=tuple(crop[:2]),sam_inventory_verified=True)
    eligible=0;high=False;socket=next(x['id'] for x in scope['anchors'] if x['kind']=='wire_entry_socket')
    for replay,record in zip(replays,view['mask_audit']):
        assert replay['record_id']==record['record_id']
        assert sorted((n,tuple(sorted(h.items()))) for n,h in replay['signatures'])==sorted((c['pixel_count'],tuple(sorted(c['anchor_pixel_support'].items()))) for c in record['components'])
        for n,h in replay['signatures']:
            if record['score']>=.75 and not record['boundary_truncated'] and all(v>0 for v in h.values()):eligible+=1
            if record['score']>=.75 and h[socket]>0:high=True
    assert eligible==len(view['eligible_native_components']) and high==view['high_score_mask_touches_socket']
    verify(protocol['pins']);assert source_pins()==protocol['mainline_pins']
    save(out/a.case/'case_audit.json',dict(status='PASS',observation=view,scalar_pixel_replay=True,exact_inverse=True,visual_review='pending',deployed=False))
    print(json.dumps(dict(case=a.case,eligible=eligible,high_socket=high)))

if __name__=='__main__':main()
