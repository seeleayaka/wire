"""Fixed source-only native SAM contrast; never deployed and never joins pixels."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from run_prompt_contrast import digest, save, verify
from bundle_runtime_pins import source_pins

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/sam_prompt_confidence_source_20261008'
PROMPTS = ['cable', 'wire harness', 'electrical wire']


def prepare():
    import numpy as np
    from PIL import Image
    old = ROOT / 'artifacts/mendeley_cable_socket_controls_20261006'
    p = json.loads((old / 'protocol.json').read_text(encoding='utf-8'))
    prepared = json.loads((old / 'preparation_report.json').read_text(encoding='utf-8'))
    assert source_pins() == p['mainline_pins']
    OUT.mkdir(exist_ok=False)
    pins = {str(Path(__file__).resolve()): digest(__file__),
            p['confirmed_scope_path']: digest(p['confirmed_scope_path']),
            p['checkpoint']: digest(p['checkpoint'])}
    rows = []
    for case in p['cases']:
        row = next(r for r in prepared['cases'] if r['id'] == case['id'])
        assert digest(case['original_source']['path']) == case['original_source']['image_sha256']
        fresh = Image.open(case['original_source']['path']).convert('RGB').crop(case['crop_box_xyxy'])
        assert np.array_equal(np.asarray(fresh), np.asarray(Image.open(case['source']['path']).convert('RGB')))
        path = OUT / (case['id'] + '_fresh_crop.png')
        fresh.save(path)
        pins[case['original_source']['path']] = case['original_source']['image_sha256']
        pins[str(path)] = digest(path)
        rows.append(dict(case, fresh_crop=str(path), anchors=row['anchors'],
                         socket_state=row['socket_evidence']['socket_state'] if 'socket_state' in row['socket_evidence']
                         else row.get('phenotype', 'uncertain')))
    processor = Path(p['sam_source']) / 'sam3/model/sam3_image_processor.py'
    pins[str(processor)] = digest(processor)
    save(OUT / 'protocol.json', dict(cases=rows, prompts=PROMPTS, retrieval_threshold=.3,
        comparison_thresholds=[.3, .5, .75], component_score_min=.75, pins=pins,
        mainline_pins=p['mainline_pins'], sam_source=p['sam_source'], checkpoint=p['checkpoint'],
        confirmed_scope_path=p['confirmed_scope_path'], model_observer_count=1,
        fresh_original_crop=True, reuse_only_registered_anchors=True,
        no_demo_images=True, no_deployment=True, no_mask_repair=True,
        hypothesis='prompt may improve native connectivity; retrieval filtering does not repair pixels',
        gate='strict source gain; retain old support; no added exposed-socket conflict; otherwise reject'))
    print('prepared five fixed source controls')


def run():
    os.environ['HF_HUB_OFFLINE'] = '1'
    p = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    verify(p['pins'])
    assert source_pins() == p['mainline_pins']
    assert not (OUT / 'progress.json').exists(), 'never restart an existing run'
    sys.path.insert(0, p['sam_source'])
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    torch.set_num_threads(8)
    torch.manual_seed(0)
    started = time.monotonic()
    completed = []
    save(OUT / 'progress.json', dict(status='building_model', pid=os.getpid(), completed=[]))
    try:
        model = build_sam3_image_model(device='cpu', checkpoint_path=p['checkpoint'],
            load_from_HF=False, enable_inst_interactivity=False, compile=False)
        processor = Sam3Processor(model, resolution=1008, device='cpu', confidence_threshold=.3)
        for case in p['cases']:
            assert source_pins() == p['mainline_pins']
            image = Image.open(case['fresh_crop']).convert('RGB')
            save(OUT / 'progress.json', dict(status='fresh_encoder', case=case['id'], pid=os.getpid(), completed=completed))
            with torch.amp.autocast('cpu', dtype=torch.bfloat16):
                state = processor.set_image(image)
            for prompt in p['prompts']:
                processor.reset_all_prompts(state)
                save(OUT / 'progress.json', dict(status='decoding', case=case['id'], prompt=prompt,
                                               pid=os.getpid(), completed=completed))
                for boxed in [False, True]:
                    with torch.amp.autocast('cpu', dtype=torch.bfloat16):
                        result = (processor.add_geometric_prompt(case['positive_box_cxcywh_normalized'], True, state)
                                  if boxed else processor.set_text_prompt(prompt=prompt, state=state))
                    masks = result['masks'].squeeze(1).detach().cpu().numpy().astype(bool)
                    scores = result['scores'].detach().float().cpu().numpy()
                    boxes = result['boxes'].detach().float().cpu().numpy()
                    assert len(masks) == len(scores) == len(boxes) and (scores > .3).all()
                    name = prompt.replace(' ', '_') + ('_box' if boxed else '_text')
                    dest = OUT / case['id'] / name
                    dest.mkdir(parents=True)
                    np.savez_compressed(dest / 'native.npz', masks=masks, scores=scores, boxes=boxes)
                    for i, mask in enumerate(masks):
                        Image.fromarray(mask.astype('uint8') * 255).save(dest / ('mask_%03d.png' % i))
                    save(dest / 'inventory.json', dict(prompt=prompt, boxed=boxed,
                        collection_threshold=.3, scores=scores.tolist(), boxes=boxes.tolist(),
                        threshold_counts={str(t): int((scores > t).sum()) for t in p['comparison_thresholds']},
                        native_sha256=digest(dest / 'native.npz'), model_observer_count=1,
                        encoder_from_current_run=True, geometry_reset_before_each_text=True))
                    print(case['id'], name, 'masks', len(scores), flush=True)
            completed.append(case['id'])
            if time.monotonic() - started > 5400:
                raise TimeoutError('90 minute bounded source experiment')
        verify(p['pins'])
        assert source_pins() == p['mainline_pins']
        report = dict(status='complete', completed=completed, fresh_encoders=len(completed),
                      fresh_decoders=len(completed)*6, seconds=time.monotonic()-started,
                      source_pins_unchanged=True, deployed=False, electrical_connections_confirmed=0)
        save(OUT / 'inference_report.json', report)
        save(OUT / 'progress.json', report)
    except BaseException as e:
        save(OUT / 'progress.json', dict(status='failed', error=str(e), completed=completed))
        raise


def audit():
    import numpy as np
    from PIL import Image, ImageDraw
    from audit_semantic_visible_bundle_native import flood_components
    from audit_cable_socket_controls import replay_hits
    p = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    assert json.loads((OUT / 'inference_report.json').read_text())['status'] == 'complete'
    verify(p['pins'])
    scope = json.loads(Path(p['confirmed_scope_path']).read_text(encoding='utf-8'))
    socket_id = next(a['id'] for a in scope['anchors'] if a['kind'] == 'wire_entry_socket')
    rows = []
    for case in p['cases']:
        for prompt in p['prompts']:
            for boxed in [False, True]:
                name = prompt.replace(' ', '_') + ('_box' if boxed else '_text')
                dest = OUT / case['id'] / name
                inventory = json.loads((dest / 'inventory.json').read_text())
                assert digest(dest / 'native.npz') == inventory['native_sha256']
                arrays = np.load(dest / 'native.npz', allow_pickle=False)
                audits = []
                image = Image.open(case['fresh_crop']).convert('RGB')
                sheet = Image.new('RGB', (400*3, 440*max(1, (len(arrays['scores'])+2)//3)), 'white')
                draw = ImageDraw.Draw(sheet)
                for i, (mask, score) in enumerate(zip(arrays['masks'], arrays['scores'])):
                    assert np.array_equal(mask, np.asarray(Image.open(dest / ('mask_%03d.png' % i))) > 0)
                    ys, xs = np.where(mask)
                    h, w = mask.shape
                    boundary = bool(len(xs) and (xs.min() <= 1 or ys.min() <= 1 or xs.max() >= w-2 or ys.max() >= h-2))
                    hits = replay_hits(flood_components(mask), w, case['crop_box_xyxy'][:2], scope['anchors'], case['anchors'])
                    native_both = sum(all(v > 0 for v in item.values()) for _, item in hits)
                    eligible = native_both if score >= .75 and not boundary else 0
                    high_socket = bool(score >= .75 and any(item[socket_id] > 0 for _, item in hits))
                    audits.append(dict(mask=i, score=float(score), boundary=boundary,
                        native_both_anchor_components=native_both, eligible=eligible, high_socket_touch=high_socket,
                        components=[dict(pixels=n, hits=hit) for n, hit in hits]))
                    rgb = np.asarray(image).copy()
                    rgb[mask] = (rgb[mask]*.45 + np.array([0,200,255])*.55).astype('uint8')
                    x, y = (i % 3)*400, (i//3)*440
                    sheet.paste(Image.fromarray(rgb).resize((400,400)), (x,y))
                    draw.text((x+5,y+405), '%s %.4f eligible=%d' % (i,float(score),eligible), fill='black')
                sheet.save(dest / 'all_masks.png')
                rows.append(dict(id=case['id'], prompt=prompt, boxed=boxed,
                    masks=len(audits), threshold_counts=inventory['threshold_counts'],
                    eligible=sum(a['eligible'] for a in audits),
                    high_socket_touch=any(a['high_socket_touch'] for a in audits), audit=audits))
    assert source_pins() == p['mainline_pins']
    save(OUT / 'report.json', dict(status='complete', cases=rows, component_score_min=.75,
        all_native_masks_retained=True, same_model_observer_count=1, deployed=False,
        electrical_connections_confirmed=0, actual_visual_review='pending',
        not_field_accuracy=True, no_mask_bridging=True))
    print(json.dumps([{k:v for k,v in r.items() if k != 'audit'} for r in rows]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','run','audit'])
    mode = parser.parse_args().mode
    globals()[mode]()
