"""Native output replay independent of SAM; one case, no mask repair."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256
from run_prompt_contrast import save,verify
from run_audit import source_pins
from run_paired_evidence import load_run
from run_review import verified_run
from analyze_mendeley_scope import render_all
from audit_semantic_visible_bundle_native import flood_components
from audit_cable_socket_controls import replay_hits

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',required=True);args=parser.parse_args()
    out=ROOT/'artifacts/mendeley_paired_endpoint_controls_20261007'
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    verify(protocol['pins']);assert source_pins()==protocol['mainline_pins']
    row=next(c for c in protocol['cases'] if c['id']==args.case)
    scope=json.loads(Path(protocol['confirmed_scope_path']).read_text(encoding='utf-8'))
    origin=row['original_source'];crop=row['crop_context']['crop_box_xyxy'];image_path=row['crop_context']['source']['path']
    assert sha256(origin['path'])==origin['image_sha256']
    actual=np.asarray(Image.open(origin['path']).convert('RGB').crop(crop))
    assert np.array_equal(actual,np.asarray(Image.open(image_path).convert('RGB')))
    run=out/args.case/protocol['acquisition_recipe'];source,records=load_run(run);verified_run(run,image_path)
    render_all(source,records,out/(args.case+'_native_outputs'))
    socket=next(a['id'] for a in scope['anchors'] if a['kind']=='wire_entry_socket')
    eligible=[];high=False;audit=[]
    for record in records:
        raw=np.asarray(Image.open(record['source_mask_path']).convert('L'))>0
        h,w=raw.shape;assert raw.shape==actual.shape[:2]
        parts=flood_components(raw);ys,xs=np.where(raw)
        boundary=bool(len(xs) and (xs.min()<=1 or ys.min()<=1 or xs.max()>=w-2 or ys.max()>=h-2))
        assert boundary==record['boundary_truncated']
        signatures=replay_hits(parts,w,crop[:2],scope['anchors'],row['anchors'])
        components=[]
        for index,(n,hits) in enumerate(signatures,1):
            good=bool(record['score']>=.75 and not boundary and all(v>0 for v in hits.values()))
            if good:eligible.append(dict(mask=record['record_id'],component=index))
            if record['score']>=.75 and hits[socket]>0:high=True
            components.append(dict(native_pixel_count=n,anchor_pixel_support=hits,eligible=good))
        audit.append(dict(mask=record['record_id'],score=record['score'],whole_mask_boundary=boundary,
            source_mask_sha256=sha256(record['source_mask_path']),components=components))
    verify(protocol['pins']);assert source_pins()==protocol['mainline_pins']
    save(out/(args.case+'_audit.json'),dict(status='PASS',id=args.case,phenotype=row['phenotype'],native_masks=len(records),
        eligible_native_components=eligible,unique_native_two_anchor_component=len(eligible)==1,high_socket_touch=high,
        audit=audit,independent_flood_fill_scalar_projection=True,actual_visual_review='pending',
        all_components_and_masks_retained=True,electrical_connections_confirmed=0,deployed=False))
    print(json.dumps(dict(id=args.case,unique_native_two_anchor_component=len(eligible)==1,high_socket_touch=high)))

if __name__=='__main__':main()
