"""One isolated fresh SAM text+anatomy-box experiment, same observer not two votes."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from run_prompt_contrast import digest, save, verify, write_masks

sys.dont_write_bytecode=True
os.environ['HF_HUB_OFFLINE']='1'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args();protocol=json.loads(args.protocol.read_text(encoding='utf-8'));out=args.protocol.parent
    if (out/'progress.json').exists():raise FileExistsError('do not repeat started experiment')
    verify(protocol['pins']);protocol_hash=digest(args.protocol)
    if (protocol['recipes']!=['cable','cable_plus_reference_anatomy_box']
            or protocol['retrieval_threshold']!=.5 or protocol['whole_mask_geometry_threshold']!=.75):
        raise ValueError('frozen recipe drift')
    begun=time.perf_counter();completed=[]
    try:
        save(out/'progress.json',{'status':'building_model','pid':os.getpid(),'completed':[]})
        sys.path.insert(0,protocol['sam_source'])
        import numpy as np
        import torch
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        torch.set_num_threads(8);torch.manual_seed(0)
        start=time.perf_counter()
        model=build_sam3_image_model(device='cpu',checkpoint_path=protocol['checkpoint'],
            load_from_HF=False,enable_inst_interactivity=False,compile=False)
        build=time.perf_counter()-start
        processor=Sam3Processor(model,resolution=1008,device='cpu',confidence_threshold=.5)
        for case in protocol['cases']:
            if time.perf_counter()-begun>1800:raise TimeoutError('between-case30min limit')
            source=case['source'];image=Image.open(source['path']).convert('RGB')
            if digest(source['path'])!=source['image_sha256']:raise ValueError('source drift')
            save(out/'progress.json',{'status':'fresh_encoder','case':case['id'],'pid':os.getpid(),
                'completed':completed,'seconds':time.perf_counter()-begun})
            with torch.amp.autocast('cpu',dtype=torch.bfloat16):
                start=time.perf_counter();state=processor.set_image(image);encoder=time.perf_counter()-start
                start=time.perf_counter();result=processor.set_text_prompt(state=state,prompt='cable');decoder=time.perf_counter()-start
            for recipe in protocol['recipes']:
                if recipe.endswith('_box'):
                    save(out/'progress.json',{'status':'geometric_decoder','case':case['id'],'pid':os.getpid(),
                        'completed':completed,'seconds':time.perf_counter()-begun})
                    with torch.amp.autocast('cpu',dtype=torch.bfloat16):
                        start=time.perf_counter()
                        result=processor.add_geometric_prompt(case['positive_box_cxcywh_normalized'],True,state)
                        decoder=time.perf_counter()-start
                masks=result['masks'].squeeze(1).detach().cpu().numpy().astype(bool)
                boxes=result['boxes'].detach().float().cpu().numpy();scores=result['scores'].detach().float().cpu().numpy()
                run=out/case['id']/recipe
                write_masks(run,image,masks,boxes,scores,recipe,encoder,decoder,source,protocol['pins'],build)
                save(run/'geometry_prompt_provenance.json',{'recipe':recipe,
                    'box':case['positive_box_cxcywh_normalized'] if recipe.endswith('_box') else None,
                    'original_source':case['original_source'],'crop_box_xyxy':case['crop_box_xyxy'],
                    'reference_only_anatomy_selection':True,'same_model_observer_count':1})
                print(json.dumps({'case':case['id'],'recipe':recipe,'masks':len(scores)},ensure_ascii=False),flush=True)
                del masks,boxes,scores,result
            del state,image
            completed.append(case['id'])
        verify(protocol['pins'])
        if digest(args.protocol)!=protocol_hash:raise ValueError('protocol drift')
        report={'status':'complete','completed':completed,'fresh_encoders':len(completed),
            'fresh_decoders':len(completed)*2,'seconds':time.perf_counter()-begun,
            'protocol_sha256':protocol_hash,'source_pins_unchanged':True,
            'new_confirmed_electrical_connections':0,'model_observer_count':1,'deployed':False}
        save(out/'inference_report.json',report);save(out/'progress.json',report)
    except BaseException as error:
        save(out/'progress.json',{'status':'failed','error':str(error),'type':type(error).__name__,'completed':completed})
        (out/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
