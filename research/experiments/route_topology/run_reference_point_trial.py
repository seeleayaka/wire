"""One fresh SAM3 interactive instance decoder; IoU quality, not cable confidence."""
import os
import sys
import json
import time
from pathlib import Path
from run_prompt_contrast import digest,save,verify
from bundle_runtime_pins import source_pins

ROOT=Path(__file__).resolve().parents[2]


def main():
    output=ROOT/'artifacts/mendeley_reference_point_trial_20261006'
    if (output/'progress.json').exists():raise FileExistsError('trial already started')
    path=output/'protocol.json';p=json.loads(path.read_text(encoding='utf-8'))
    verify(p['pins']);protocol_hash=digest(path)
    start=time.perf_counter()
    save(output/'progress.json',{'status':'building_interactive_model','pid':os.getpid()})
    try:
        os.environ['HF_HUB_OFFLINE']='1';sys.path.insert(0,p['sam_source'])
        import numpy as np
        import torch
        from PIL import Image
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        torch.set_num_threads(8);torch.manual_seed(0)
        model=build_sam3_image_model(device='cpu',checkpoint_path=p['checkpoint'],load_from_HF=False,
            enable_inst_interactivity=True,compile=False)
        if model.inst_interactive_predictor is None:raise ValueError('interactive weights not available')
        checkpoint=torch.load(p['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
        if 'model' in checkpoint and isinstance(checkpoint['model'],dict):checkpoint=checkpoint['model']
        loaded={k.replace('detector.','') for k in checkpoint if 'detector' in k}
        loaded.update(k.replace('tracker.','inst_interactive_predictor.model.') for k in checkpoint if 'tracker' in k)
        missing=set(model.state_dict())-loaded
        save(output/'weight_loading_audit.json',{'expected_keys':len(model.state_dict()),
            'missing_keys':sorted(missing),'interactive_weights_checked_before_encoder':True})
        del checkpoint
        if missing:raise ValueError('missing pretrained interactive model weights; no inference permitted')
        processor=Sam3Processor(model,resolution=1008,device='cpu',confidence_threshold=.5)
        with Image.open(p['source']['path']) as im:image=im.convert('RGB')
        save(output/'progress.json',{'status':'fresh_encoder','pid':os.getpid()})
        with torch.inference_mode(),torch.amp.autocast('cpu',dtype=torch.bfloat16):
            begun=time.perf_counter();state=processor.set_image(image);encoder=time.perf_counter()-begun
            save(output/'progress.json',{'status':'interactive_decoder','pid':os.getpid()})
            begun=time.perf_counter()
            masks,qualities,logits=model.predict_inst(state,
                point_coords=np.asarray(p['point_coords_crop'],np.float32),point_labels=np.asarray(p['point_labels'],np.int32),
                box=np.asarray(p['box_xyxy_crop'],np.float32),multimask_output=p['multimask_output'])
            decoder=time.perf_counter()-begun
        if masks.shape!=(len(qualities),image.height,image.width) or not np.isfinite(qualities).all():
            raise ValueError('interactive output frame/quality invalid')
        records=[]
        for i,(mask,quality) in enumerate(zip(masks,qualities),1):
            active=(mask>0).astype(np.uint8)
            file=output/f'native_mask_{i:03d}.png';Image.fromarray(active*255).save(file)
            records.append({'path':str(file),'sha256':digest(file),'quality_prediction':float(quality),
                            'score_type':p['score_type'],'pixels':int(active.sum())})
        # No filtering, clipping of scores, mask postprocessing, or synthetic links.
        verify(p['pins'])
        if digest(path)!=protocol_hash or source_pins()!=p['mainline_pins']:raise ValueError('protocol/E drift')
        report={'status':'complete','fresh_encoders':1,'interactive_decoders':1,'native_outputs':records,
            'seconds':time.perf_counter()-start,'encoder_seconds':encoder,'decoder_seconds':decoder,
            'protocol_sha256':protocol_hash,'score_type':p['score_type'],'model_observer_count':1,
            'points_are_prompts_not_evidence':True,'reference_feasibility_gate':'pending','deployed':False,
            'electrical_correctness':'not_assessed'}
        save(output/'inference_report.json',report);save(output/'progress.json',report)
    except BaseException as error:
        save(output/'progress.json',{'status':'failed','error':str(error),'do_not_auto_retry':True})
        raise


if __name__=='__main__':main()
