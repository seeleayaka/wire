"""Reference feasibility only; actual whole-harness native masks, no relabeling."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from run_review import read, save, verified_run
from run_prompt_contrast import verify
from core import sha256, extract_mask
from run_paired_evidence import load_run
from analyze_mendeley_scope import render_all
from bundle_runtime_pins import source_pins

ROOT = Path(__file__).resolve().parents[2]


def eligible_components(raw, score, anchors, translation):
    geometry = extract_mask(raw, score, 'reference_harness')
    n, labels = cv2.connectedComponents((raw > 0).astype(np.uint8), connectivity=8)
    eligible, audits = [], []
    for index in range(1, n):
        ys, xs = np.nonzero(labels == index)
        x, y = xs + translation[0], ys + translation[1]
        hits = {}
        for anchor in anchors:
            l, t, r, b = anchor['bbox_xyxy']
            hits[anchor['id']] = int(np.count_nonzero((x >= l) & (x <= r) & (y >= t) & (y <= b)))
        accepted = bool(score >= .75 and not geometry['boundary_truncated'] and all(hits.values()))
        if accepted:
            eligible.append(index)
        audits.append({'component_index': index, 'pixels': len(xs), 'hits': hits, 'eligible': accepted})
    return eligible, audits


def main():
    folder = ROOT / 'artifacts/mendeley_harness_reference_trial_20261006'
    p = read(folder / 'protocol.json')
    inference = read(folder / 'inference_report.json')
    if inference['status'] != 'complete' or inference['protocol_sha256'] != sha256(folder / 'protocol.json'):
        raise ValueError('fresh inference incomplete or protocol drift')
    verify(p['pins'])
    if source_pins() != p['mainline_pins']:
        raise ValueError('E drift')
    scope = read(p['confirmed_scope_path'])
    output = ROOT / 'artifacts/mendeley_harness_reference_audit_20261006'
    output.mkdir(exist_ok=False)
    rows = []
    for recipe in p['recipes']:
        run = folder / recipe
        inventory = verified_run(run, p['source']['path'])
        eligible = []
        audits = []
        for index, (path, score) in enumerate(zip(inventory['paths'], inventory['scores'])):
            with Image.open(path) as opened:
                raw = np.asarray(opened.convert('L'))
            good, parts = eligible_components(raw, score, scope['anchors'], p['crop_box_xyxy'][:2])
            eligible += [(index + 1, part) for part in good]
            audits.append({'mask_index': index + 1, 'score': score, 'components': parts})
        origin, records = load_run(run)
        render_all(origin, records, output / recipe)
        rows.append({'recipe': recipe, 'eligible_native_components': eligible,
                     'component_audits': audits, 'reference_feasible': len(eligible) == 1})
    verify(p['pins'])
    save(output / 'report.json', {'status': 'complete', 'cases': rows,
        'reference_boxed_feasibility_passed': rows[-1]['reference_feasible'],
        'actual_visual_review': 'pending', 'independent_arithmetic_replay': 'pending',
        'source_controls_not_run': True, 'deployed': False,
        'electrical_correctness': 'not_assessed', 'model_observer_count': 1,
        'next_if_failed': 'retain failure; no prompt sweep or inspection batch',
        'pins': {str(f): sha256(f) for f in [Path(__file__), folder/'protocol.json', folder/'inference_report.json']}})
    print({'reference_boxed_feasibility_passed': rows[-1]['reference_feasible']})


if __name__ == '__main__':
    main()
