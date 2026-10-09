"""Single fixed acquisition view; failed reference/source stops remaining controls."""
import argparse,json,os,sys,time,subprocess,traceback
from pathlib import Path
from run_prompt_contrast import digest,save,verify,write_masks
sys.dont_write_bytecode=True
os.environ['HF_HUB_OFFLINE']='1'

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);a=p.parse_args()
    protocol=json.loads(a.protocol.read_text(encoding='utf-8'));out=a.protocol.parent
    if (out/'progress.json').exists():raise FileExistsError('no duplicate/retry')
    verify(protocol['pins']);ph=digest(a.protocol);completed=[];rows=[];begun=time.perf_counter();stop=None
    assert protocol['recipes']==['cable','cable_plus_reference_anatomy_box']
    assert protocol['retrieval_threshold']==.5 and protocol['whole_mask_geometry_threshold']==.75
    try:
        save(out/'progress.json',dict(status='building_model',pid=os.getpid(),completed=[]))
        sys.path.insert(0,protocol['sam_source'])
        import numpy as np
        import torch
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        torch.set_num_threads(8);torch.manual_seed(0);start=time.perf_counter()
        model=build_sam3_image_model(device='cpu',checkpoint_path=protocol['checkpoint'],load_from_HF=False,enable_inst_interactivity=False,compile=False)
        build=time.perf_counter()-start;processor=Sam3Processor(model,resolution=1008,device='cpu',confidence_threshold=.5)
        for case in protocol['cases']:
            if time.perf_counter()-begun>1800:raise TimeoutError('between-case30min limit')
            source=case['source'];assert digest(source['path'])==source['image_sha256']
            image=Image.open(source['path']).convert('RGB')
            save(out/'progress.json',dict(status='fresh_encoder',case=case['id'],pid=os.getpid(),completed=completed))
            with torch.amp.autocast('cpu',dtype=torch.bfloat16):
                start=time.perf_counter();state=processor.set_image(image);enc=time.perf_counter()-start
                start=time.perf_counter();result=processor.set_text_prompt(state=state,prompt='cable');dec=time.perf_counter()-start
            for recipe in protocol['recipes']:
                if recipe.endswith('_box'):
                    with torch.amp.autocast('cpu',dtype=torch.bfloat16):
                        start=time.perf_counter();result=processor.add_geometric_prompt(case['positive_box_cxcywh_normalized'],True,state);dec=time.perf_counter()-start
                masks=result['masks'].squeeze(1).detach().cpu().numpy().astype(bool)
                boxes=result['boxes'].detach().float().cpu().numpy();scores=result['scores'].detach().float().cpu().numpy()
                write_masks(out/case['id']/recipe,image,masks,boxes,scores,recipe,enc,dec,source,protocol['pins'],build)
                print(json.dumps(dict(case=case['id'],recipe=recipe,masks=len(scores))),flush=True)
                del masks,boxes,scores,result
            del state,image
            save(out/'progress.json',dict(status='independent_native_replay',case=case['id'],pid=os.getpid(),completed=completed))
            subprocess.run(['E:/PythonProject10/.venv/Scripts/python.exe','-B',str(Path(__file__).with_name('audit_quarter_turn_case.py')),'--protocol',str(a.protocol),'--case',case['id']],check=True)
            audit=json.loads((out/case['id']/'case_audit.json').read_text(encoding='utf-8'));assert audit['status']=='PASS'
            completed.append(case['id']);rows.append(dict(id=case['id'],observation=audit['observation']))
            if case['id'] in ['reference','source_visible_01'] and not audit['observation']['unique_native_bundle_observation_supported']:
                stop='reference_native_component_failed' if case['id']=='reference' else 'visible01_no_new_native_gain'
                break
        verify(protocol['pins']);assert digest(a.protocol)==ph
        from run_audit import source_pins
        assert source_pins()==protocol['mainline_pins']
        views={r['id']:r['observation'] for r in rows}
        gate=(len(completed)==5 and all(views[k]['unique_native_bundle_observation_supported'] for k in ['reference','source_visible_01','source_visible_02']) and not any(views[k]['high_score_mask_touches_socket'] for k in ['source_exposed_01','source_exposed_02']))
        report=dict(status='complete',completed=completed,unexecuted=[k for k in protocol['all_five_planned_ids'] if k not in completed],cases=rows,stop_reason=stop,source_gate_passed=gate,fresh_encoders=len(completed),fresh_decoders=2*len(completed),seconds=time.perf_counter()-begun,protocol_sha256=ph,source_pins_unchanged=True,visual_review='pending',deployed=False,electrical_connections_confirmed=0,model_observer_count=1)
        save(out/'report.json',report);save(out/'progress.json',report)
    except BaseException as error:
        save(out/'progress.json',dict(status='failed',error=str(error),completed=completed,no_retry=True))
        (out/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
