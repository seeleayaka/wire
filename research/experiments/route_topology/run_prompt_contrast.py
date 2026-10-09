"""Frozen local SAM3 prompt contrast. Fresh image encoders; no topology claim.

Runs in the existing SAM environment. Never consumes historical pickle states.
All prompts on an image share ONE fresh backbone and ONE model observer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
os.environ['HF_HUB_OFFLINE'] = '1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def verify(pins):
    for path, expected in pins.items():
        if digest(path) != expected:
            raise ValueError('frozen input or source changed: ' + path)


def write_masks(run, image, masks, boxes, scores, prompt, encoder_seconds,
                decoder_seconds, source, runtime, model_build_seconds):
    import numpy as np
    from PIL import Image
    sam = run / 'sam'
    sam.mkdir(parents=True, exist_ok=False)
    snapshot = run / ('source_snapshot' + Path(source['path']).suffix)
    shutil.copyfile(source['path'], snapshot)
    if digest(snapshot) != source['image_sha256']:
        raise ValueError('snapshot/source drift')
    if masks.shape != (len(scores), image.height, image.width):
        raise ValueError('mask shape/frame mismatch')
    if boxes.shape != (len(scores), 4) or not np.isfinite(scores).all() or not np.isfinite(boxes).all():
        raise ValueError('invalid model outputs')
    if not ((scores > .5) & (scores <= 1)).all():
        raise ValueError('unexpected retrieval score')
    for index, mask in enumerate(masks, 1):
        Image.fromarray(mask.astype(np.uint8) * 255).save(sam / f'mask_{index:03d}.png')
    union = np.any(masks, axis=0) if len(masks) else np.zeros((image.height, image.width), bool)
    Image.fromarray(union.astype(np.uint8) * 255).save(sam / 'mask_union.png')
    report = {'input':str(snapshot.resolve()), 'prompt':prompt,
              'confidence_threshold':.5, 'device':'cpu', 'threads':8,
              'instance_count':len(scores), 'scores':scores.tolist(),
              'boxes_xyxy':boxes.tolist(), 'mask_pixel_counts':[int(m.sum()) for m in masks],
              'image_state_cache_reused':False, 'fresh_encoder_for_source':True,
              'shared_fresh_encoder_between_prompts':True, 'model_observer_count':1,
              'timings':{'model_build_seconds':model_build_seconds,
                         'image_encoder_seconds':encoder_seconds,
                         'text_and_mask_seconds':decoder_seconds}}
    save(sam / 'report.json', report)
    files = [snapshot, sam / 'report.json', sam / 'mask_union.png',
             *[p for p in sorted(sam.glob('mask_*.png')) if p.stem[5:].isdigit()]]
    binding = {k:source[k] for k in ['image_sha256','image_size','coordinate_frame']}
    save(run / 'run_manifest.json', {'schema_version':1, 'status':'complete',
         'run_id':run.name, 'evidence_kind':'fresh_local_sam_prompt_contrast',
         'image_binding':dict(binding, image_path=source['path']),
         'runtime_fingerprints':runtime,
         'recipe':{'prompt':prompt, 'threshold':.5, 'threads':8,
                   'no_registration':True, 'no_cache_reuse':True,
                   'shared_fresh_encoder_between_prompts':True, 'model_observer_count':1},
         'verified_files':{str(p.resolve()):digest(p) for p in files},
         'automatic_fault_verdict':False, 'observation_coverage_confirmed':False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    out = args.protocol.parent
    protocol_hash = digest(args.protocol)
    verify(protocol['pins'])
    if protocol['prompts'] != ['cable','wire'] or protocol['retrieval_threshold'] != .5:
        raise ValueError('unsupported frozen recipe')
    if (out / 'progress.json').exists():
        raise FileExistsError('do not repeat a started experiment')
    start = time.perf_counter()
    completed = []
    try:
        save(out / 'progress.json', {'status':'building_model','pid':os.getpid(), 'completed_cases':[]})
        sys.path.insert(0, str(Path(protocol['sam_source'])))
        import numpy as np
        import torch
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        torch.set_num_threads(8)
        torch.manual_seed(0)
        begun = time.perf_counter()
        model = build_sam3_image_model(device='cpu', checkpoint_path=protocol['checkpoint'],
                load_from_HF=False, enable_inst_interactivity=False, compile=False)
        build_seconds = time.perf_counter() - begun
        processor = Sam3Processor(model, resolution=1008, device='cpu', confidence_threshold=.5)
        for case in protocol['cases']:
            if time.perf_counter() - start > 1800:
                raise TimeoutError('30 minute between-case budget exceeded')
            source = case['source']
            if digest(source['path']) != source['image_sha256']:
                raise ValueError('source drift')
            with Image.open(source['path']) as opened:
                image = opened.convert('RGB')
            if list(image.size) != source['image_size']:
                raise ValueError('source dimensions drift')
            save(out / 'progress.json', {'status':'fresh_image_encoder','case':case['id'],
                 'pid':os.getpid(),'completed_cases':completed,'elapsed_seconds':time.perf_counter()-start})
            with torch.amp.autocast('cpu', dtype=torch.bfloat16):
                begun = time.perf_counter()
                state = processor.set_image(image)
                encoder_seconds = time.perf_counter() - begun
            for prompt in protocol['prompts']:
                processor.reset_all_prompts(state)  # In-place; returns None.
                save(out / 'progress.json', {'status':'text_decoder','case':case['id'],
                     'prompt':prompt,'pid':os.getpid(),'completed_cases':completed,
                     'elapsed_seconds':time.perf_counter()-start})
                with torch.amp.autocast('cpu', dtype=torch.bfloat16):
                    begun = time.perf_counter()
                    result = processor.set_text_prompt(state=state, prompt=prompt)
                    decoder_seconds = time.perf_counter() - begun
                masks = result['masks'].squeeze(1).detach().cpu().numpy().astype(bool)
                boxes = result['boxes'].detach().float().cpu().numpy()
                scores = result['scores'].detach().float().cpu().numpy()
                write_masks(out / case['id'] / prompt, image, masks, boxes, scores, prompt,
                            encoder_seconds, decoder_seconds, source, protocol['pins'], build_seconds)
                print(json.dumps({'case':case['id'],'prompt':prompt,'masks':len(scores),
                      'encoder_seconds':round(encoder_seconds,2)},ensure_ascii=False),flush=True)
                del masks, boxes, scores, result
            del state, image
            completed.append(case['id'])
        verify(protocol['pins'])
        if digest(args.protocol) != protocol_hash:
            raise ValueError('protocol drift')
        report = {'status':'complete','cases_completed':completed,'fresh_image_encoders':len(completed),
                  'new_prompt_decoders':len(completed)*2,'seconds':time.perf_counter()-start,
                  'protocol_sha256':protocol_hash,'pins_unchanged':True,
                  'model_observer_count':1,'new_confirmed_real_connections':0,
                  'automatic_fault_verdict':False,'E_deployed':False}
        save(out / 'inference_report.json', report)
        save(out / 'progress.json', report)
    except BaseException as exc:
        save(out / 'progress.json', {'status':'failed','error_type':type(exc).__name__,
             'error':str(exc),'completed_cases':completed,'seconds':time.perf_counter()-start})
        (out / 'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
