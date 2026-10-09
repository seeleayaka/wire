"""All-mask frozen prompt experiment audit; no semantic/terminal GT inferred."""
import json
from pathlib import Path
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from core import extract_mask
from run_audit import source_pins
from run_paired_evidence import load_run
from run_prompt_contrast import digest, save, verify

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/route_prompt_contrast_20261005'


def audit_run(path):
    origin, records = load_run(path)
    arrays = [np.asarray(Image.open(item['path']).convert('L')) > 0 for item in origin['mask_files']]
    reasons = Counter()
    eligible = []
    for record in records:
        rejected = []
        if record['score'] < .75:
            rejected.append('score_below_0.75')
        if record['boundary_truncated']:
            rejected.append('native_frame_boundary')
        if record['geometry']['state'] != 'simple_visible_path':
            rejected.append(record['geometry']['state'])
        record['geometry_evidence_eligible'] = not rejected
        record['rejections'] = rejected
        reasons.update(rejected)
        if not rejected:
            eligible.append(record['record_id'])
    return origin, records, arrays, {'all_masks':len(records), 'eligible_ids':eligible,
            'eligible_count':len(eligible), 'rejection_reasons_nonexclusive':dict(reasons),
            'confirmed_real_connections':0}


def overlaps(first, second):
    """Per-mask maxima only; not one-to-one matching, truth or accuracy."""
    values = []
    for mask in first:
        pair = []
        for target in second:
            union = int(np.logical_or(mask,target).sum())
            pair.append(float(np.logical_and(mask,target).sum()/union) if union else 0.)
        values.append({'best_target_id':int(np.argmax(pair))+1 if pair else None,
                       'best_IoU':max(pair,default=0.)})
    return values


def contact_sheet(image_path, records, arrays, path):
    with Image.open(image_path) as opened:
        original = opened.convert('RGB')
    tile_w, tile_h = 300, 345
    cols = 4
    rows = max(1,(len(records)+cols-1)//cols)
    sheet = Image.new('RGB',(cols*tile_w,rows*tile_h),'#f7f8fa')
    font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf',14)
    for index,(record,mask) in enumerate(zip(records,arrays)):
        pixels = np.asarray(original,dtype=np.float32).copy()
        pixels[mask] = .45*pixels[mask]+.55*np.array([30,160,210])
        tile = Image.fromarray(pixels.astype(np.uint8))
        draw = ImageDraw.Draw(tile)
        for x,y in record['geometry']['tips_xy']:
            draw.ellipse((x-2,y-2,x+2,y+2),fill='#ff3300')
        tile.thumbnail((tile_w-8,tile_h-65))
        x,y = (index%cols)*tile_w,(index//cols)*tile_h
        sheet.paste(tile,(x+4,y+4))
        draw = ImageDraw.Draw(sheet)
        draw.text((x+4,y+tile_h-57),f"{record['record_id']} {record['score']:.3f}",fill='black',font=font)
        draw.text((x+4,y+tile_h-37),record['geometry']['state'],fill='black',font=font)
        draw.text((x+4,y+tile_h-17),'geometry eligible' if record['geometry_evidence_eligible'] else ', '.join(record['rejections']),fill='#994422',font=font)
    sheet.save(path)


def main():
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    inference = json.loads((OUT/'inference_report.json').read_text(encoding='utf-8'))
    if inference['status']!='complete' or digest(OUT/'protocol.json')!=inference['protocol_sha256']:
        raise ValueError('requires completed bound inference')
    verify(protocol['pins'])
    if source_pins()!=protocol['mainline_pins']:
        raise ValueError('mainline or dirty worktree changed')
    cases=[]
    for case in protocol['cases']:
        old,oldrecords,oldarrays,oldsummary=audit_run(case['historical_run'])
        if old['manifest_sha256']!=case['historical_manifest_sha256']:
            raise ValueError('historical drift')
        cable,crecords,carrays,csummary=audit_run(OUT/case['id']/'cable')
        wire,wrecords,warrays,wsummary=audit_run(OUT/case['id']/'wire')
        oldsignatures=sorted((r['mask_array_sha256'],r['score']) for r in oldrecords)
        newsignatures=sorted((r['mask_array_sha256'],r['score']) for r in crecords)
        backward=overlaps(carrays,warrays)
        forward=overlaps(warrays,carrays)
        detail={'case':case['id'],'historical':oldsummary,'fresh_cable':csummary,
                'fresh_wire':wsummary,'fresh_cable_exact_reproduction':oldsignatures==newsignatures,
                'cable_to_wire_best_overlap':backward,'wire_to_cable_best_overlap':forward,
                'eligible_cable_overlaps':[dict(record_id=r['record_id'],**backward[i])
                     for i,r in enumerate(crecords) if r['geometry_evidence_eligible']],
                'new_geometry_evidence_not_physical_connections':True,
                'cable_records':crecords,'wire_records':wrecords,
                'all_semantics_or_port_identities_confirmed':False}
        save(OUT/case['id']/'geometry_report.json',detail)
        for prompt,records,arrays in [('cable',crecords,carrays),('wire',wrecords,warrays)]:
            contact_sheet(case['source']['path'],records,arrays,OUT/case['id']/f'{prompt}_all_masks.png')
        cases.append({k:detail[k] for k in ['case','historical','fresh_cable','fresh_wire',
                     'fresh_cable_exact_reproduction','eligible_cable_overlaps']})
    verify(protocol['pins'])
    if source_pins()!=protocol['mainline_pins']:
        raise ValueError('mainline drift during analysis')
    save(OUT/'report.json',{'status':'complete','cases':cases,'inference':inference,
         'manual_semantics_audit':'pending','minimum_score':.75,'model_observer_count':1,
         'new_confirmed_real_connections':0,'new_true_hit_gain':None,'physical_false_alarms':None,
         'field_accuracy':None,'topology_decision':'insufficient_evidence',
         'mainline_pins_and_dirty_status_unchanged':True,'E_deployed':False})
    print(json.dumps(cases,ensure_ascii=False))


if __name__ == '__main__':
    sys.dont_write_bytecode=True
    main()
