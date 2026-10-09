"""One frozen exposed-source counterexample; no acceptance gate relaxation.

Reference outputs remain the declared prior fresh probe, not a second inference.
This run measures whether the same plug+anatomy prompt also masks socket housing.
"""
import json
import os
from pathlib import Path
import sys
import time
import traceback
from run_prompt_contrast import digest, save, verify, write_masks

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True
os.environ['HF_HUB_OFFLINE']='1'

def main():
    base=ROOT/'artifacts/mendeley_plug_source_probe_v2_20261006'
    out=ROOT/'artifacts/mendeley_plug_negative_diagnostic_20261006'
    out.mkdir(exist_ok=False)
    protocol=json.loads((base/'protocol.json').read_text(encoding='utf-8'))
    case=next(c for c in protocol['cases'] if c['id']=='source_exposed_01')
    pins=dict(protocol['pins'])
    pins[str(Path(__file__).resolve())]=digest(__file__)
    verify(pins)
    save(out/'protocol.json',dict(case=case, pins=pins, prompt='plug',
         recipes=protocol['recipes'], retrieval_threshold=.5,
         purpose='negative source semantic discrimination diagnostic only',
         reference_evidence=str(base/'reference'),
         reference_reused_explicitly=True, no_acceptance_rule_change=True,
         no_mask_trimming_or_bridging=True, deployed=False))
    started=time.monotonic()
    try:
        save(out/'progress.json',dict(status='building_model',pid=os.getpid()))
        sys.path.insert(0,protocol['sam_source'])
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        torch.set_num_threads(8);torch.manual_seed(0)
        t=time.monotonic()
        model=build_sam3_image_model(device='cpu',checkpoint_path=protocol['checkpoint'],
                load_from_HF=False,enable_inst_interactivity=False,compile=False)
        build=time.monotonic()-t
        processor=Sam3Processor(model,resolution=1008,device='cpu',confidence_threshold=.5)
        source=case['source']
        assert digest(source['path'])==source['image_sha256']
        image=Image.open(source['path']).convert('RGB')
        original=Image.open(case['original_source']['path']).convert('RGB')
        assert digest(case['original_source']['path'])==case['original_source']['image_sha256']
        assert np.array_equal(np.asarray(image),np.asarray(original.crop(case['crop_box_xyxy'])))
        save(out/'progress.json',dict(status='fresh_encoder',pid=os.getpid(),case=case['id']))
        with torch.amp.autocast('cpu',dtype=torch.bfloat16):
            t=time.monotonic();state=processor.set_image(image);encoder=time.monotonic()-t
        rows=[]
        for recipe in protocol['recipes']:
            save(out/'progress.json',dict(status='decoder',recipe=recipe,pid=os.getpid()))
            with torch.amp.autocast('cpu',dtype=torch.bfloat16):
                t=time.monotonic()
                if recipe=='plug':result=processor.set_text_prompt(state=state,prompt='plug')
                else:result=processor.add_geometric_prompt(case['positive_box_cxcywh_normalized'],True,state)
                decoder=time.monotonic()-t
            masks=result['masks'].squeeze(1).detach().cpu().numpy().astype(bool)
            boxes=result['boxes'].detach().float().cpu().numpy()
            scores=result['scores'].detach().float().cpu().numpy()
            write_masks(out/case['id']/recipe,image,masks,boxes,scores,recipe,encoder,decoder,source,pins,build)
            objects=[]
            for index,(mask,score) in enumerate(zip(masks,scores),1):
                n,labels,stats,centroids=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
                objects.append(dict(mask_id=index,score=float(score),native_components=n-1,
                     component_areas=stats[1:,4].tolist(),mask_pixels=int(mask.sum())))
            rows.append(dict(recipe=recipe,objects=objects))
            print(json.dumps(rows[-1]),flush=True)
        verify(pins)
        report=dict(status='complete',case=case['id'],recipes=rows,
             fresh_encoders=1,fresh_decoders=2,reference_not_rerun=True,
             seconds=time.monotonic()-started,semantic_identity_review_required=True,
             no_acceptance_rule_change=True,physical_connection_verified=False,deployed=False)
        save(out/'report.json',report);save(out/'progress.json',dict(status='complete'))
    except BaseException as error:
        save(out/'progress.json',dict(status='failed',error=str(error)))
        (out/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise

if __name__=='__main__':main()
