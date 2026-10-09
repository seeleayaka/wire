"""Independent pixel-set flood fill for the reference-only harness gate."""
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from run_review import read, save, verified_run
from run_prompt_contrast import verify
from core import sha256
from bundle_runtime_pins import source_pins

ROOT = Path(__file__).resolve().parents[2]


def parts(active):
    remaining = set(zip(*np.nonzero(active)))
    result = []
    while remaining:
        seed = remaining.pop()
        stack, component = [seed], {seed}
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (y+dy, x+dx)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
                        component.add(neighbor)
        result.append(component)
    return result


def main():
    source = ROOT / 'artifacts/mendeley_harness_reference_trial_20261006'
    audit = ROOT / 'artifacts/mendeley_harness_reference_audit_20261006/report.json'
    p, report = read(source/'protocol.json'), read(audit)
    verify(p['pins'])
    verify(report['pins'])
    if source_pins() != p['mainline_pins'] or [r['recipe'] for r in report['cases']] != p['recipes']:
        raise ValueError('E drift or recipe inventory differs')
    scope = read(p['confirmed_scope_path'])
    total = 0
    results = []
    for row in report['cases']:
        native = verified_run(source/row['recipe'], p['source']['path'])
        if len(native['paths']) != len(row['component_audits']):
            raise ValueError('mask inventory changed')
        eligible = 0
        for path, score, mask in zip(native['paths'], native['scores'], row['component_audits']):
            if score != mask['score']:
                raise ValueError('native score differs')
            with Image.open(path) as im:
                active = np.asarray(im.convert('L')) > 0
            height, width = active.shape
            boundary = any(y <= 1 or x <= 1 or y >= height-2 or x >= width-2
                           for y, x in zip(*np.nonzero(active)))
            actual = []
            for component in parts(active):
                hits = {}
                for anchor in scope['anchors']:
                    l, t, r, b = anchor['bbox_xyxy']
                    hits[anchor['id']] = sum(l <= x+p['crop_box_xyxy'][0] <= r
                          and t <= y+p['crop_box_xyxy'][1] <= b for y,x in component)
                good = bool(mask['score'] >= .75 and not boundary and all(hits.values()))
                actual.append((len(component), tuple(sorted(hits.items())), good))
                eligible += int(good)
                total += 1
            expected = [(c['pixels'], tuple(sorted(c['hits'].items())), c['eligible'])
                        for c in mask['components']]
            if Counter(actual) != Counter(expected):
                raise ValueError('independent pixel/component gate differs')
        if eligible != len(row['eligible_native_components']) or (eligible == 1) != row['reference_feasible']:
            raise ValueError('reference feasibility differs')
        results.append({'recipe': row['recipe'], 'reference_feasible': eligible == 1})
    output = ROOT / 'artifacts/mendeley_harness_reference_replay_20261006'
    output.mkdir(exist_ok=False)
    save(output/'report.json', {'status': 'PASS', 'native_components_replayed': total,
        'cases': results, 'reference_boxed_feasibility_passed': results[-1]['reference_feasible'],
        'not_field_accuracy': True, 'source_controls_not_run': True, 'deployed': False,
        'pins': {str(f): sha256(f) for f in [Path(__file__), audit, source/'protocol.json']}})
    print({'independent_replay': 'PASS', 'reference_boxed_feasibility': results[-1]['reference_feasible']})


if __name__ == '__main__':
    main()
