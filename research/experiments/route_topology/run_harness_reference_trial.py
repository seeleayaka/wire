"""One preregistered whole-harness prompt on the reviewed reference only.

Evidence collection, not a topology decision or an accepted replacement recipe.
The fresh original crop and encoder do not reuse earlier SAM embeddings/masks.
"""
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timezone
from run_prompt_contrast import verify, write_masks, save, digest as sha256
from bundle_runtime_pins import source_pins

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare():
    old = ROOT / 'artifacts/mendeley_socket_extent_source_controls_20261006'
    if read(old / 'pipeline_progress.json')['status'] != 'complete':
        raise ValueError('protect active source pipeline; do not start concurrent SAM')
    p = read(old / 'protocol.json')
    verify(p['pins'])
    if source_pins() != p['mainline_pins']:
        raise ValueError('E mainline drift')
    ref = next(c for c in p['cases'] if c['id'] == 'reference')
    output = ROOT / 'artifacts/mendeley_harness_reference_trial_20261006'
    output.mkdir(exist_ok=False)
    from PIL import Image
    with Image.open(ref['original_source']['path']) as opened:
        image = opened.convert('RGB').crop(ref['crop_box_xyxy'])
    image.save(output / 'fresh_original_crop.png')
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'reviewed reference only; never inspection photo selection',
        'prompt': 'wire harness', 'recipes': ['wire_harness', 'wire_harness_plus_reference_anatomy_box'],
        'retrieval_threshold': .5, 'geometry_score_threshold': .75,
        'source': {'path': str(output / 'fresh_original_crop.png'),
                   'image_sha256': sha256(output / 'fresh_original_crop.png'),
                   'image_size': list(image.size), 'coordinate_frame': 'source_image_pixels'},
        'original_source': ref['original_source'], 'crop_box_xyxy': ref['crop_box_xyxy'],
        'box': ref['positive_box_cxcywh_normalized'],
        'checkpoint': p['checkpoint'], 'sam_source': p['sam_source'],
        'confirmed_scope_path': p['confirmed_scope_path'], 'mainline_pins': p['mainline_pins'],
        'pins': {**p['pins'], str(Path(__file__).resolve()): sha256(__file__),
                 str(output / 'fresh_original_crop.png'): sha256(output / 'fresh_original_crop.png')},
        'model_observer_count': 1, 'fresh_encoder_required': True,
        'no_mask_bridging_or_geometry_relaxation': True, 'deployed': False,
        'official_API_reference': 'https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image_processor.py',
        'hypothesis': 'Whole-harness wording may retrieve native bundle regions including wrap; no guarantee.',
        'stop_if': 'no unique native reference component touches both reviewed anchors at score .75; no prompt sweep',
        'source_gate_required_before_inspection_batch': True,
        'electrical_correctness': 'not_assessed'}
    save(output / 'protocol.json', protocol)
    return output, protocol


def main():
    output, p = prepare()
    started = time.perf_counter()
    save(output / 'progress.json', {'status': 'building_model', 'pid': os.getpid()})
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        sys.path.insert(0, p['sam_source'])
        import torch
        import numpy as np
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        torch.set_num_threads(8)
        torch.manual_seed(0)
        start = time.perf_counter()
        model = build_sam3_image_model(device='cpu', checkpoint_path=p['checkpoint'],
                    load_from_HF=False, enable_inst_interactivity=False, compile=False)
        build_seconds = time.perf_counter() - start
        processor = Sam3Processor(model, resolution=1008, device='cpu', confidence_threshold=.5)
        with Image.open(p['source']['path']) as im:
            image = im.convert('RGB')
        save(output / 'progress.json', {'status': 'fresh_encoder', 'pid': os.getpid()})
        with torch.amp.autocast('cpu', dtype=torch.bfloat16):
            start = time.perf_counter()
            state = processor.set_image(image)
            encoder = time.perf_counter() - start
        inventory = []
        for recipe in p['recipes']:
            with torch.amp.autocast('cpu', dtype=torch.bfloat16):
                start = time.perf_counter()
                if recipe == 'wire_harness':
                    result = processor.set_text_prompt(state=state, prompt=p['prompt'])
                else:
                    result = processor.add_geometric_prompt(p['box'], True, state)
                decoder = time.perf_counter() - start
            masks = result['masks'].squeeze(1).detach().cpu().numpy().astype(bool)
            boxes = result['boxes'].detach().float().cpu().numpy()
            scores = result['scores'].detach().float().cpu().numpy()
            write_masks(output / recipe, image, masks, boxes, scores, recipe,
                        encoder, decoder, p['source'], p['pins'], build_seconds)
            save(output / recipe / 'geometry_prompt_provenance.json', {
                'recipe': recipe, 'literal_text_prompt': p['prompt'],
                'box': p['box'] if recipe.endswith('_box') else None,
                'same_model_observer_count': 1, 'reference_only': True})
            inventory.append({'recipe': recipe, 'masks': len(scores)})
        verify(p['pins'])
        if source_pins() != p['mainline_pins']:
            raise ValueError('E drift')
        report = {'status': 'complete', 'fresh_encoders': 1, 'fresh_decoders': 2,
            'seconds': time.perf_counter()-started, 'inventory': inventory,
            'reference_feasibility_gate': 'pending', 'actual_visual_review': 'pending',
            'protocol_sha256': sha256(output / 'protocol.json'),
            'deployed': False, 'electrical_correctness': 'not_assessed'}
        save(output / 'inference_report.json', report)
        save(output / 'progress.json', report)
    except BaseException as error:
        save(output / 'progress.json', {'status': 'failed', 'error': str(error),
            'do_not_auto_retry': True})
        raise


if __name__ == '__main__':
    main()
