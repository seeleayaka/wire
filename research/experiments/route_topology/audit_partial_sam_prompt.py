"""Fast independent vectorized native component audit; explicitly partial."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from run_prompt_contrast import digest, save

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/sam_prompt_confidence_source_20261008'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    case_id = parser.parse_args().case
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    case = next(r for r in protocol['cases'] if r['id'] == case_id)
    scope = json.loads(Path(protocol['confirmed_scope_path']).read_text(encoding='utf-8'))
    rows = []
    for dest in sorted((OUT / case_id).iterdir()):
        if not (dest / 'inventory.json').exists(): continue
        inv = json.loads((dest / 'inventory.json').read_text())
        assert digest(dest / 'native.npz') == inv['native_sha256']
        data = np.load(dest / 'native.npz', allow_pickle=False)
        records = []
        image = Image.open(case['fresh_crop']).convert('RGB')
        width, height = image.size
        canvas = Image.new('RGB', (width*3, (height+40)*max(1,(len(data['scores'])+2)//3)), 'white')
        draw = ImageDraw.Draw(canvas)
        for i,(mask,score) in enumerate(zip(data['masks'],data['scores'])):
            assert np.array_equal(mask,np.asarray(Image.open(dest / ('mask_%03d.png' % i))) > 0)
            n,labels,stats,_ = cv2.connectedComponentsWithStats(mask.astype('uint8'),connectivity=8)
            ys,xs = np.where(mask)
            boundary = bool(len(xs) and (xs.min()<=1 or ys.min()<=1 or xs.max()>=width-2 or ys.max()>=height-2))
            parts = []
            for c in range(1,n):
                ys,xs = np.where(labels==c)
                points = np.column_stack((xs+case['crop_box_xyxy'][0],ys+case['crop_box_xyxy'][1],np.ones(len(xs))))
                hits = {}
                for anchor in scope['anchors']:
                    pose = next(p for p in case['anchors'] if p['id']==anchor['id'])
                    hit = 0
                    if pose['localization_proposal_supported'] and all(pose['gates'].values()):
                        q = points @ np.asarray(pose['inspection_to_reference_local']).T
                        assert (np.abs(q[:,2]) > 1e-9).all()
                        q = q[:,:2]/q[:,2:]
                        l,t,r,b = anchor['bbox_xyxy']
                        hit = int(((q[:,0]>=l)&(q[:,0]<=r)&(q[:,1]>=t)&(q[:,1]<=b)).sum())
                    hits[anchor['id']] = hit
                parts.append(dict(pixels=int(stats[c,cv2.CC_STAT_AREA]), hits=hits,
                    eligible=bool(score>=.75 and not boundary and all(v>0 for v in hits.values()))))
            eligible = sum(p['eligible'] for p in parts)
            records.append(dict(score=float(score), boundary=boundary, eligible=eligible, components=parts))
            rgb = np.asarray(image).copy()
            rgb[mask] = (rgb[mask]*.45 + np.array([0,200,255])*.55).astype('uint8')
            x,y = (i%3)*width,(i//3)*(height+40)
            canvas.paste(Image.fromarray(rgb),(x,y))
            draw.text((x+3,y+height+5),'mask %d %.4f eligible=%d' % (i,float(score),eligible),fill='black')
        canvas.save(dest / 'partial_all_masks.png')
        rows.append(dict(recipe=dest.name,threshold_counts=inv['threshold_counts'],
                         eligible=sum(r['eligible'] for r in records),records=records))
    save(OUT / (case_id+'_partial_audit.json'),dict(status='partial',case=case_id,rows=rows,
        actual_visual_review='pending',electrical_connections_confirmed=0,deployed=False))
    print(json.dumps([{k:v for k,v in r.items() if k!='records'} for r in rows]))


if __name__ == '__main__': main()
