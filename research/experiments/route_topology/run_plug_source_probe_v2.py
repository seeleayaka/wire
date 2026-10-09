"""Fixed plug noun + socket anatomy; preserve masks, fail reference before extension."""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from run_prompt_contrast import save,verify,digest,write_masks

os.environ['HF_HUB_OFFLINE']='1';sys.dont_write_bytecode=True
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'));out=args.protocol.parent
    if (out/'progress.json').exists():raise FileExistsError('do not duplicate started source trial')
    if protocol['recipes']!=['plug','plug_plus_socket_anatomy_box'] or protocol['retrieval_threshold']!=.5:raise ValueError('fixed recipe drift')
    verify(protocol['pins']);protocol_sha=digest(args.protocol);start=time.monotonic();rows=[]
    try:
        save(out/'progress.json',dict(status='building_model',pid=os.getpid(),completed=[]))
        sys.path.insert(0,protocol['sam_source'])
        import numpy as np
        import torch
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        import cv2
        torch.set_num_threads(8);torch.manual_seed(0)
        begun=time.monotonic();model=build_sam3_image_model(device='cpu',checkpoint_path=protocol['checkpoint'],load_from_HF=False,enable_inst_interactivity=False,compile=False)
        build=time.monotonic()-begun;processor=Sam3Processor(model,resolution=1008,device='cpu',confidence_threshold=.5)
        for case in protocol['cases']:
            if time.monotonic()-start>1800:raise TimeoutError('source trial30min between-case budget')
            source=case['source']
            if digest(source['path'])!=source['image_sha256']:raise ValueError('original crop drift')
            image=Image.open(source['path']).convert('RGB')
            save(out/'progress.json',dict(status='fresh_encoder',case=case['id'],pid=os.getpid(),completed=[r['id'] for r in rows],seconds=time.monotonic()-start))
            with torch.amp.autocast('cpu',dtype=torch.bfloat16):
                t=time.monotonic();state=processor.set_image(image);encoder=time.monotonic()-t
                t=time.monotonic();result=processor.set_text_prompt(state=state,prompt='plug');decoder=time.monotonic()-t
            nominations=[];mask_count={}
            for recipe in protocol['recipes']:
                if recipe.endswith('_box'):
                    with torch.amp.autocast('cpu',dtype=torch.bfloat16):
                        t=time.monotonic();result=processor.add_geometric_prompt(case['positive_box_cxcywh_normalized'],True,state);decoder=time.monotonic()-t
                masks=result['masks'].squeeze(1).detach().cpu().numpy().astype(bool)
                boxes=result['boxes'].detach().float().cpu().numpy();scores=result['scores'].detach().float().cpu().numpy()
                write_masks(out/case['id']/recipe,image,masks,boxes,scores,recipe,encoder,decoder,source,protocol['pins'],build)
                mask_count[recipe]=len(scores)
                if recipe.endswith('_box'):
                    l,t,r,b=np.asarray(case['positive_box_source_xyxy'])-np.asarray(case['crop_box_xyxy'][:2]*2)
                    for index,(raw,score) in enumerate(zip(masks,scores)):
                        count,_=cv2.connectedComponents(raw.astype(np.uint8),connectivity=8)
                        ys,xs=np.where(raw)
                        truncated=bool(len(xs) and (xs.min()<=1 or ys.min()<=1 or xs.max()>=raw.shape[1]-2 or ys.max()>=raw.shape[0]-2))
                        if score>=.75 and not truncated and count-1==1 and raw[max(0,t):b,max(0,l):r].any():
                            nominations.append(f'mask_{index+1:03d}')
                print(json.dumps(dict(case=case['id'],recipe=recipe,masks=len(scores))),flush=True)
                del result,masks,boxes,scores
            rows.append(dict(id=case['id'],mask_count=mask_count,plug_region_nominations=nominations,
                unique_plug_region_nomination=len(nominations)==1,plug_semantic_identity_or_insertion_verified=False))
            del state,image
            if case['id']=='reference' and len(nominations)!=1:break
        verify(protocol['pins'])
        if digest(args.protocol)!=protocol_sha:raise ValueError('protocol drift')
        report=dict(status='complete',cases=rows,fresh_encoders=len(rows),fresh_decoders=2*len(rows),
            reference_nomination_gate=rows[0]['unique_plug_region_nomination'],
            all_fixed_sources_executed=len(rows)==len(protocol['cases']),visual_semantic_review_pending=True,
            protocol_sha256=protocol_sha,seconds=time.monotonic()-start,model_observer_count=1,
            no_mask_pixels_changed=True,not_a_physical_occupancy_or_topology_verdict=True,deployed=False)
        save(out/'report.json',report);save(out/'progress.json',dict(status='complete',completed=[r['id'] for r in rows]))
        print(json.dumps(report),flush=True)
    except BaseException as error:
        save(out/'progress.json',dict(status='failed',error=str(error),completed=[r['id'] for r in rows]))
        (out/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()


