"""Fresh native two-endpoint acquisition; early reference failure stops expansion."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from run_prompt_contrast import save,verify,digest,write_masks

ROOT=Path(__file__).resolve().parents[2]

def main():
    out=ROOT/'artifacts/mendeley_paired_endpoint_controls_20261007'
    protocol_path=out/'protocol.json';p=json.loads(protocol_path.read_text(encoding='utf-8'))
    if (out/'progress.json').exists():raise FileExistsError('no retry or duplicate worker')
    verify(p['pins']);protocol_digest=digest(protocol_path);started=time.perf_counter();rows=[];termination='all_source_controls_processed'
    save(out/'progress.json',dict(status='building_model',pid=os.getpid(),completed=[]))
    try:
        sys.path.insert(0,p['sam_source']);os.environ['HF_HUB_OFFLINE']='1'
        import numpy as np
        import torch
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        torch.set_num_threads(8);torch.manual_seed(0)
        model_started=time.perf_counter()
        model=build_sam3_image_model(device='cpu',checkpoint_path=p['checkpoint'],load_from_HF=False,
            enable_inst_interactivity=False,compile=False)
        build=time.perf_counter()-model_started
        processor=Sam3Processor(model,resolution=1008,device='cpu',confidence_threshold=.5)
        for row in p['cases']:
            if time.perf_counter()-started>1800:raise TimeoutError('between-case30min budget')
            context=row['crop_context'];source=context['source']
            assert digest(source['path'])==source['image_sha256']
            image=Image.open(source['path']).convert('RGB')
            save(out/'progress.json',dict(status='fresh_encoder',case=row['id'],pid=os.getpid(),completed=[r['id'] for r in rows]))
            with torch.amp.autocast('cpu',dtype=torch.bfloat16):
                at=time.perf_counter();state=processor.set_image(image);encoder=time.perf_counter()-at
                at=time.perf_counter();result=processor.set_text_prompt(state=state,prompt='cable')
                for index,box in enumerate(row['positive_endpoint_boxes_cxcywh_normalized'],1):
                    save(out/'progress.json',dict(status='endpoint_decoder',case=row['id'],endpoint=index,pid=os.getpid(),completed=[r['id'] for r in rows]))
                    result=processor.add_geometric_prompt(box,True,state)
                decoder=time.perf_counter()-at
            masks=result['masks'].squeeze(1).detach().cpu().numpy().astype(bool)
            boxes=result['boxes'].detach().float().cpu().numpy();scores=result['scores'].detach().float().cpu().numpy()
            run=out/row['id']/p['acquisition_recipe']
            write_masks(run,image,masks,boxes,scores,p['acquisition_recipe'],encoder,decoder,source,p['pins'],build)
            save(run/'endpoint_prompt_provenance.json',dict(endpoint_boxes=row['positive_endpoint_boxes_cxcywh_normalized'],
                original_source=row['original_source'],crop_box_xyxy=context['crop_box_xyxy'],
                location_prompts_not_connection_evidence=True,same_model_observer_count=1,no_mask_union=True))
            del masks,boxes,scores,result,state,image
            audit=subprocess.run(['E:/PythonProject10/.venv/Scripts/python.exe','-B',
                str(Path(__file__).with_name('audit_paired_endpoint_case.py')),'--case',row['id']],cwd=ROOT)
            assert audit.returncode==0,'native/source replay failed; no retry'
            result=json.loads((out/(row['id']+'_audit.json')).read_text(encoding='utf-8'));rows.append(result)
            if row['id']=='reference' and not result['unique_native_two_anchor_component']:
                termination='reference_native_component_gate_failed';break
        baseline=json.loads(Path(p['baseline_report']).read_text(encoding='utf-8'))
        old={r['id']:r['observation']['unique_native_bundle_observation_supported'] for r in baseline['cases']}
        visible=[r for r in rows if r['phenotype']=='mating_body_visible']
        gains=[r['id'] for r in visible if r['unique_native_two_anchor_component'] and not old[r['id']]]
        losses=[r['id'] for r in visible if not r['unique_native_two_anchor_component'] and old[r['id']]]
        conflicts=[r['id'] for r in rows if r['phenotype']=='socket_contacts_exposed' and r['high_socket_touch']]
        passed=bool(len(rows)==5 and gains and not losses and not conflicts)
        verify(p['pins']);assert digest(protocol_path)==protocol_digest
        save(out/'report.json',dict(status='complete',termination=termination,cases=rows,
            planned_control_ids=[r['id'] for r in p['cases']],unexecuted_control_ids=[r['id'] for r in p['cases'][len(rows):]],
            source_gate_passed=passed,new_source_gains=gains,old_source_losses=losses,exposed_socket_conflicts=conflicts,
            fresh_encoders=len(rows),fresh_decoders=3*len(rows),seconds=time.perf_counter()-started,
            actual_visual_review='pending',no_demo_extension=True,electrical_connections_confirmed=0,deployed=False))
        save(out/'progress.json',dict(status='complete',source_gate_passed=passed,termination=termination))
    except BaseException as error:
        save(out/'progress.json',dict(status='failed',error=str(error),completed=[r['id'] for r in rows],no_retry=True));raise

if __name__=='__main__':main()
