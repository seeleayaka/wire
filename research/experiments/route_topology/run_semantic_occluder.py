"""Fresh full-context semantic obstacle proposals; inferred bundle paths only."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import traceback
from run_prompt_contrast import digest, save, verify, write_masks

sys.dont_write_bytecode=True
os.environ['HF_HUB_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/semantic_occluder_source_20261008'
PROMPTS=['cable tie','label','black cable']


def semantic_edges(ends, edges, raw, objects):
    from occluder_tangent_candidates import witness
    from soft_tangent_candidates import PARAMS
    result=deepcopy(edges);by_id={e['endpoint_id']:e for e in ends}
    ranking={i:[] for i in by_id}
    for i,e in enumerate(result):
        e['state']='unaccepted_semantic_obstacle_edge'
        e['occluder_witness']=witness(e,by_id,raw,objects)
        if e['occluder_witness']['candidate'] and e['score']<=PARAMS['max_score']:
            for k in e['endpoint_ids']:ranking[k].append((e['score'],i))
    choices={}
    for k,rows in ranking.items():
        rows.sort();choices[k]=rows[0][1] if rows and (len(rows)==1 or rows[1][0]-rows[0][0]>=PARAMS['margin']) else None
    for i,e in enumerate(result):
        if all(choices[k]==i for k in e['endpoint_ids']):e['state']='occluder_supported_fragment_candidate'
        e['observed_gap_pixels']=0;e['physical_identity_confirmed']=False
    return result


def prepare():
    from bundle_runtime_pins import source_pins
    p=json.loads((ROOT/'artifacts/soft_tangent_source_20261008/protocol.json').read_text(encoding='utf-8'))
    control=json.loads((ROOT/'artifacts/mendeley_cable_socket_controls_20261006/protocol.json').read_text(encoding='utf-8'))
    before=source_pins();assert before==p['mainline_pins'];pins=dict(p['pins']);pins.update(control['pins'])
    files=[Path(__file__),ROOT/'artifacts/SEMANTIC_OCCLUDER_PROTOCOL_20261008.md',
        ROOT/'artifacts/soft_tangent_source_20261008/report.json',
        ROOT/'experiments/route_topology/occluder_tangent_candidates.py',
        ROOT/'experiments/route_topology/run_occluder_tangent_source.py']
    pins.update({str(f):digest(f) for f in files});verify(pins);OUT.mkdir(exist_ok=False)
    old=json.loads(files[2].read_text(encoding='utf-8'));cases=[]
    from PIL import Image
    for row in old['cases']:
        source=row['source_binding'];assert digest(source['path'])==source['image_sha256']
        image=Image.open(source['path']).convert('RGB').crop(row['crop_box_xyxy'])
        path=OUT/(row['id']+'_original_crop.png');image.save(path)
        pins[str(path)]=digest(path)
        cases.append(dict(id=row['id'],original=row,source=dict(path=str(path),image_sha256=digest(path),
            image_size=list(image.size),coordinate_frame='original_crop_pixels')))
    save(OUT/'protocol.json',dict(pins=pins,mainline_pins=before,cases=cases,prompts=PROMPTS,
        sam_source=control['sam_source'],checkpoint=control['checkpoint'],acceptance_threshold=.75,
        model_observer_count=1,geometry_score=.5,margin=.12,no_demo_extension=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare();return
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));verify(p['pins']);ph=digest(OUT/'protocol.json')
    assert p['prompts']==PROMPTS and p['acceptance_threshold']==.75
    if (OUT/'progress.json').exists():raise FileExistsError('no retry')
    begun=time.perf_counter();completed=[]
    try:
        save(OUT/'progress.json',dict(status='building_model',pid=os.getpid(),completed=[]))
        sys.path.insert(0,p['sam_source'])
        import numpy as np
        import torch
        from PIL import Image,ImageDraw
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        from soft_tangent_candidates import observe_family
        from run_occluder_tangent_source import paths
        from run_reference_color_paths_source import exact_regions
        from bundle_runtime_pins import source_pins
        bp=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
        poses={c['id']:c for c in json.loads((bp/'preparation_report.json').read_text(encoding='utf-8'))['cases']}
        cp=json.loads((bp/'protocol.json').read_text(encoding='utf-8'))
        scope=json.loads(Path(cp['confirmed_scope_path']).read_text(encoding='utf-8'))
        torch.set_num_threads(8);torch.manual_seed(0);start=time.perf_counter()
        model=build_sam3_image_model(device='cpu',checkpoint_path=p['checkpoint'],load_from_HF=False,enable_inst_interactivity=False,compile=False)
        build=time.perf_counter()-start;processor=Sam3Processor(model,resolution=1008,device='cpu',confidence_threshold=.5);rows=[]
        for case in p['cases']:
            if time.perf_counter()-begun>2700:raise TimeoutError('45 minute between-case ceiling')
            source=case['source'];old=case['original'];assert digest(source['path'])==source['image_sha256']
            assert digest(old['source_binding']['path'])==old['source_binding']['image_sha256']
            image=Image.open(source['path']).convert('RGB');rgb=np.asarray(image)
            save(OUT/'progress.json',dict(status='fresh_encoder',case=case['id'],pid=os.getpid(),completed=completed))
            with torch.amp.autocast('cpu',dtype=torch.bfloat16):
                start=time.perf_counter();state=processor.set_image(image);encoder=time.perf_counter()-start
            objects=[];seen=set();inventory=[]
            for prompt in PROMPTS:
                processor.reset_all_prompts(state)
                save(OUT/'progress.json',dict(status='semantic_decoder',case=case['id'],prompt=prompt,pid=os.getpid(),completed=completed))
                with torch.amp.autocast('cpu',dtype=torch.bfloat16):
                    start=time.perf_counter();result=processor.set_text_prompt(state=state,prompt=prompt);decoder=time.perf_counter()-start
                masks=result['masks'].squeeze(1).detach().cpu().numpy().astype(bool)
                boxes=result['boxes'].detach().float().cpu().numpy();scores=result['scores'].detach().float().cpu().numpy()
                directory=OUT/case['id']/prompt.replace(' ','_')
                write_masks(directory,image,masks,boxes,scores,prompt,encoder,decoder,source,p['pins'],build)
                for index,(mask,score) in enumerate(zip(masks,scores),1):
                    path=directory/'sam'/f'mask_{index:03d}.png';sha=digest(path)
                    accepted=float(score)>=.75 and 24<=int(mask.sum())<=.25*mask.size
                    inventory.append(dict(prompt=prompt,score=float(score),mask_path=str(path),mask_sha256=sha,accepted=accepted))
                    if not accepted or sha in seen:continue
                    seen.add(sha);yy,xx=np.nonzero(mask)
                    objects.append(dict(kind=prompt,component=len(objects)+1,pixels=int(mask.sum()),axis_ratio=None,
                        mask=mask,xy=np.column_stack((xx,yy))))
                del masks,boxes,scores,result
            regions=exact_regions(rgb.shape[:2],old['crop_box_xyxy'],scope,poses[case['id']]['anchors']);records=[]
            canvas=image.copy();draw=ImageDraw.Draw(canvas)
            for rec in old['records']:
                assert rec['score']>=.75 and digest(rec['mask_path'])==rec['mask_sha256']
                raw=np.asarray(Image.open(rec['mask_path']).convert('L'))>=128
                result,_=observe_family(raw,regions);edges=semantic_edges(result['endpoints'],result['edges'],raw,objects)
                oldpaths=paths(raw,[],regions);newpaths=paths(raw,edges,regions)
                assert not oldpaths or newpaths
                records.append(dict(record_id=rec['record_id'],mask_path=rec['mask_path'],mask_sha256=rec['mask_sha256'],
                    endpoints=result['endpoints'],edges=edges,old_paths=oldpaths,new_paths=newpaths))
                for e in edges:
                    if e['state']!='occluder_supported_fragment_candidate':continue
                    curve=e['virtual_curve_xy']
                    for i in range(0,len(curve)-1,4):draw.line([tuple(curve[i]),tuple(curve[min(i+1,len(curve)-1)])],fill='lime',width=2)
            canvas.save(OUT/(case['id']+'_inferred_paths.png'))
            row=dict(id=case['id'],source_binding=old['source_binding'],crop_box_xyxy=old['crop_box_xyxy'],
                original_socket_state=old['original_socket_state'],semantic_inventory=inventory,accepted_objects=len(objects),
                records=records,old_path=any(r['old_paths'] for r in records),new_path=any(r['new_paths'] for r in records))
            rows.append(row);save(OUT/case['id']/'analysis.json',row);completed.append(case['id'])
            print(json.dumps(dict(id=case['id'],accepted_objects=len(objects),old=row['old_path'],new=row['new_path'],seconds=time.perf_counter()-begun)),flush=True)
            del state,image,rgb,objects
        gains=[r['id'] for r in rows if not r['old_path'] and r['new_path']];losses=[r['id'] for r in rows if r['old_path'] and not r['new_path']]
        exposed=[r['id'] for r in rows if r['original_socket_state']=='socket_contacts_exposed' and r['new_path']]
        gate=bool(gains) and not losses and not exposed and all(r['new_path'] for r in rows if r['original_socket_state']=='mating_body_visible')
        verify(p['pins']);assert source_pins()==p['mainline_pins'];assert digest(OUT/'protocol.json')==ph
        report=dict(status='complete',cases=rows,gains=gains,losses=losses,exposed_source_candidates=exposed,source_gate_passed=gate,
            fresh_encoders=len(rows),fresh_decoders=3*len(rows),model_observer_count=1,new_confirmed_connections=0,
            physical_identity_confirmed=False,electrical_continuity='not_assessed',deployed=False,mainline_unchanged=True,seconds=time.perf_counter()-begun)
        save(OUT/'report.json',report);save(OUT/'progress.json',dict(status='complete',completed=completed,seconds=report['seconds']))
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=str(error),completed=completed,pid=os.getpid()))
        (OUT/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
