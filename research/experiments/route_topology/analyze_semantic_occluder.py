"""Regular analysis environment: SAM environment intentionally lacks skimage."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from run_prompt_contrast import digest,save,verify
from run_semantic_occluder import semantic_edges
from soft_tangent_candidates import observe_family
from run_occluder_tangent_source import paths
from run_reference_color_paths_source import exact_regions
from run_review import verified_run
from bundle_runtime_pins import source_pins

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--case');args=parser.parse_args()
    out=args.out;p=json.loads((out/'protocol.json').read_text(encoding='utf-8'));verify(p['pins'])
    if args.case:
        case=next(c for c in p['cases'] if c['id']==args.case);old=case['original'];source=case['source']
        rgb=np.asarray(Image.open(source['path']).convert('RGB'));objects=[];seen=set();inventory=[]
        for prompt in p['prompts']:
            directory=out/case['id']/prompt.replace(' ','_');verified_run(directory,Path(source['path']))
            report=json.loads((directory/'sam/report.json').read_text(encoding='utf-8'))
            for i,score in enumerate(report['scores'],1):
                path=directory/'sam'/f'mask_{i:03d}.png';sha=digest(path);mask=np.asarray(Image.open(path).convert('L'))>=128
                accepted=score>=.75 and 24<=int(mask.sum())<=.25*mask.size
                inventory.append(dict(prompt=prompt,score=score,mask_path=str(path),mask_sha256=sha,accepted=accepted))
                if not accepted or sha in seen:continue
                seen.add(sha);yy,xx=np.nonzero(mask)
                objects.append(dict(kind=prompt,component=len(objects)+1,pixels=int(mask.sum()),axis_ratio=None,mask=mask,xy=np.column_stack((xx,yy))))
        bp=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
        poses={c['id']:c for c in json.loads((bp/'preparation_report.json').read_text(encoding='utf-8'))['cases']}
        cp=json.loads((bp/'protocol.json').read_text(encoding='utf-8'));scope=json.loads(Path(cp['confirmed_scope_path']).read_text(encoding='utf-8'))
        regions=exact_regions(rgb.shape[:2],old['crop_box_xyxy'],scope,poses[case['id']]['anchors']);records=[]
        canvas=Image.fromarray(rgb.copy());draw=ImageDraw.Draw(canvas)
        for rec in old['records']:
            assert rec['score']>=.75 and digest(rec['mask_path'])==rec['mask_sha256']
            raw=np.asarray(Image.open(rec['mask_path']).convert('L'))>=128;result,_=observe_family(raw,regions)
            edges=semantic_edges(result['endpoints'],result['edges'],raw,objects);oldpaths=paths(raw,[],regions);newpaths=paths(raw,edges,regions)
            assert not oldpaths or newpaths
            records.append(dict(record_id=rec['record_id'],mask_path=rec['mask_path'],mask_sha256=rec['mask_sha256'],endpoints=result['endpoints'],edges=edges,old_paths=oldpaths,new_paths=newpaths))
            for e in edges:
                if e['state']!='occluder_supported_fragment_candidate':continue
                curve=e['virtual_curve_xy']
                for i in range(0,len(curve)-1,4):draw.line([tuple(curve[i]),tuple(curve[min(i+1,len(curve)-1)])],fill='lime',width=2)
        canvas.save(out/(case['id']+'_inferred_paths.png'))
        row=dict(id=case['id'],source_binding=old['source_binding'],crop_box_xyxy=old['crop_box_xyxy'],original_socket_state=old['original_socket_state'],
            semantic_inventory=inventory,accepted_objects=len(objects),records=records,old_path=any(r['old_paths'] for r in records),new_path=any(r['new_paths'] for r in records))
        save(out/case['id']/'analysis.json',row);print(json.dumps(dict(id=case['id'],accepted_objects=len(objects),old=row['old_path'],new=row['new_path'])))
    else:
        rows=[json.loads((out/c['id']/'analysis.json').read_text(encoding='utf-8')) for c in p['cases']]
        gains=[r['id'] for r in rows if not r['old_path'] and r['new_path']];losses=[r['id'] for r in rows if r['old_path'] and not r['new_path']]
        exposed=[r['id'] for r in rows if r['original_socket_state']=='socket_contacts_exposed' and r['new_path']]
        gate=bool(gains) and not losses and not exposed and all(r['new_path'] for r in rows if r['original_socket_state']=='mating_body_visible')
        assert source_pins()==p['mainline_pins']
        save(out/'report.json',dict(status='complete',cases=rows,gains=gains,losses=losses,exposed_source_candidates=exposed,source_gate_passed=gate,
            fresh_encoders=len(rows),fresh_decoders=3*len(rows),model_observer_count=1,new_confirmed_connections=0,physical_identity_confirmed=False,
            electrical_continuity='not_assessed',deployed=False,mainline_unchanged=True))


if __name__=='__main__':main()
