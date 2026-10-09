"""Fresh SAM encoders on frozen local crops; reference failure stops expansion."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from run_prompt_contrast import digest,save,verify,write_masks

sys.dont_write_bytecode=True
os.environ['HF_HUB_OFFLINE']='1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);args=parser.parse_args()
    p=json.loads(args.protocol.read_text(encoding='utf-8'));out=args.protocol.parent
    if (out/'progress.json').exists():raise FileExistsError('no restart/retry')
    assert p['prompt']=='cable' and p['acceptance_threshold']==.75 and p['retrieval_threshold']==.5
    verify(p['pins']);ph=digest(args.protocol);begun=time.perf_counter();completed=[]
    try:
        save(out/'progress.json',dict(status='building_model',pid=os.getpid(),completed=[]))
        sys.path.insert(0,p['sam_source'])
        import numpy as np
        import torch
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        from local_socket_crop import native_coverage
        torch.set_num_threads(8);torch.manual_seed(0);start=time.perf_counter()
        model=build_sam3_image_model(device='cpu',checkpoint_path=p['checkpoint'],load_from_HF=False,enable_inst_interactivity=False,compile=False)
        build=time.perf_counter()-start;processor=Sam3Processor(model,resolution=1008,device='cpu',confidence_threshold=.5)
        early=False
        for case in p['cases']:
            if time.perf_counter()-begun>1800:raise TimeoutError('between-case30min limit')
            source=case['source'];assert digest(source['path'])==source['image_sha256']
            image=Image.open(source['path']).convert('RGB')
            save(out/'progress.json',dict(status='fresh_encoder',case=case['id'],pid=os.getpid(),completed=completed,seconds=time.perf_counter()-begun))
            with torch.amp.autocast('cpu',dtype=torch.bfloat16):
                start=time.perf_counter();state=processor.set_image(image);encoder=time.perf_counter()-start
                start=time.perf_counter();result=processor.set_text_prompt(state=state,prompt='cable');decoder=time.perf_counter()-start
            masks=result['masks'].squeeze(1).detach().cpu().numpy().astype(bool)
            boxes=result['boxes'].detach().float().cpu().numpy();scores=result['scores'].detach().float().cpu().numpy()
            run=out/case['id']/'cable';write_masks(run,image,masks,boxes,scores,'cable',encoder,decoder,source,p['pins'],build)
            rgb=np.asarray(image);region=np.asarray(Image.open(case['CPU_region_path']).convert('L'))>0
            local=any(native_coverage(rgb,region,m,float(s))['local_two_color_coverage_candidate'] for m,s in zip(masks,scores))
            completed.append(case['id']);print(json.dumps(dict(id=case['id'],masks=len(scores),local_coverage_candidate=local,seconds=time.perf_counter()-begun)),flush=True)
            del masks,boxes,scores,result,state,image
            if case['id']=='reference' and not local:early=True;break
        verify(p['pins']);assert digest(args.protocol)==ph
        report=dict(status='complete',completed=completed,early_reference_rejection=early,fresh_encoders=len(completed),
            seconds=time.perf_counter()-begun,protocol_sha256=ph,model_observer_count=1,new_confirmed_connections=0,deployed=False)
        save(out/'inference_report.json',report);save(out/'progress.json',report)
    except BaseException as error:
        save(out/'progress.json',dict(status='failed',error=str(error),error_type=type(error).__name__,completed=completed))
        (out/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
