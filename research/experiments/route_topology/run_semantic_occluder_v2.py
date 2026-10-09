"""Separate pinned inference/analysis environments; preserves setup failure v1."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import run_semantic_occluder as recipe
from run_prompt_contrast import digest,save,verify,write_masks

sys.dont_write_bytecode=True;os.environ['HF_HUB_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/semantic_occluder_source_v2_20261008'
ANALYZER=Path(__file__).with_name('analyze_semantic_occluder.py')
ANALYSIS_PYTHON='E:/PythonProject10/.venv/Scripts/python.exe'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:
        recipe.OUT=OUT;recipe.prepare();p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
        p['pins'].update({str(f):digest(f) for f in [Path(__file__),ANALYZER]})
        p['analysis_python']=ANALYSIS_PYTHON;p['setup_failure_v1_preserved']=True;save(OUT/'protocol.json',p);return
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));verify(p['pins']);ph=digest(OUT/'protocol.json')
    assert p['prompts']==recipe.PROMPTS and p['acceptance_threshold']==.75
    if (OUT/'progress.json').exists():raise FileExistsError('no retry')
    completed=[];begun=time.perf_counter()
    try:
        save(OUT/'progress.json',dict(status='building_model',pid=os.getpid(),completed=[]))
        sys.path.insert(0,p['sam_source'])
        import torch
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        torch.set_num_threads(8);torch.manual_seed(0);start=time.perf_counter()
        model=build_sam3_image_model(device='cpu',checkpoint_path=p['checkpoint'],load_from_HF=False,enable_inst_interactivity=False,compile=False)
        build=time.perf_counter()-start;processor=Sam3Processor(model,resolution=1008,device='cpu',confidence_threshold=.5)
        for case in p['cases']:
            if time.perf_counter()-begun>2700:raise TimeoutError('45 minute between-case ceiling')
            source=case['source'];assert digest(source['path'])==source['image_sha256']
            image=Image.open(source['path']).convert('RGB')
            save(OUT/'progress.json',dict(status='fresh_encoder',case=case['id'],pid=os.getpid(),completed=completed))
            with torch.amp.autocast('cpu',dtype=torch.bfloat16):
                start=time.perf_counter();state=processor.set_image(image);encoder=time.perf_counter()-start
            for prompt in p['prompts']:
                processor.reset_all_prompts(state)
                save(OUT/'progress.json',dict(status='semantic_decoder',case=case['id'],prompt=prompt,pid=os.getpid(),completed=completed))
                with torch.amp.autocast('cpu',dtype=torch.bfloat16):
                    start=time.perf_counter();result=processor.set_text_prompt(state=state,prompt=prompt);decoder=time.perf_counter()-start
                masks=result['masks'].squeeze(1).detach().cpu().numpy().astype(bool)
                boxes=result['boxes'].detach().float().cpu().numpy();scores=result['scores'].detach().float().cpu().numpy()
                write_masks(OUT/case['id']/prompt.replace(' ','_'),image,masks,boxes,scores,prompt,encoder,decoder,source,p['pins'],build)
                del result,masks,boxes,scores
            del state,image
            save(OUT/'progress.json',dict(status='graph_analysis',case=case['id'],pid=os.getpid(),completed=completed))
            subprocess.run([ANALYSIS_PYTHON,'-B',str(ANALYZER),'--out',str(OUT),'--case',case['id']],check=True)
            completed.append(case['id'])
        subprocess.run([ANALYSIS_PYTHON,'-B',str(ANALYZER),'--out',str(OUT)],check=True)
        verify(p['pins']);assert digest(OUT/'protocol.json')==ph
        save(OUT/'progress.json',dict(status='complete',completed=completed,seconds=time.perf_counter()-begun))
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=str(error),completed=completed,pid=os.getpid()))
        (OUT/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
