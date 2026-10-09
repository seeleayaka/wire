"""TRAIN-only data geometry readiness; no model or heldout pixel reads."""
import json
import sys
import time
from pathlib import Path
from collections import Counter

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REPO = Path('E:/PythonProject10')
sys.path.insert(0, str(REPO))
from inspection_agent.optional_port_crop_review import sha
from inspection_agent.port_training_audit import parse_boxes
from inspection_agent.port_tiling import tile_windows
from port_training_multiscale_policy import positive_window

OUT = ROOT / 'artifacts/allport480_readiness_20261004'
BASE = ROOT / 'artifacts/port_training_multiscale_20261002'
DATA = REPO / 'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'


def independent_labels(ports, window, shape):
    """Replay containment and all four guards in source coordinates."""
    h, w = shape
    x, y, r, b = window
    if not (0 <= x < r <= w and 0 <= y < b <= h):
        raise ValueError('window outside source')
    result = []
    for port in ports:
        l, t, rr, bb = port['box_xyxy']
        if rr <= x or l >= r or bb <= y or t >= b:
            continue
        if l < x or rr > r or t < y or bb > b:
            return None
        if ((x > 0 and l <= x + 16) or (y > 0 and t <= y + 16)
                or (r < w and rr >= r - 16) or (b < h and bb >= b - 16)):
            return None
        result.append(dict(class_id=port['class_id'], source_box_index=port['source_box_index'],
                           box_xyxy_local=[l-x, t-y, rr-x, bb-y]))
    return sorted(result, key=lambda p: p['source_box_index'])


def plan_all(ports, shape):
    if min(shape) < 480:
        return [], [p['source_box_index'] for p in ports]
    windows, skipped = {}, []
    for seed in sorted(ports, key=lambda p: p['source_box_index']):
        plan = positive_window(seed, ports, 480, shape)
        if plan is None:
            skipped.append(seed['source_box_index'])
            continue
        labels = sorted(plan['labels'], key=lambda p: p['source_box_index'])
        assert independent_labels(ports, plan['window'], shape) == labels
        key = tuple(plan['window'])
        if key not in windows:
            windows[key] = dict(window=plan['window'], labels=labels, seed_indices=[])
        windows[key]['seed_indices'].append(seed['source_box_index'])
    return [windows[k] for k in sorted(windows)], skipped


def legacy_labels(ports, window):
    """Old frozen training intentionally includes visible clipped rectangles."""
    x, y, r, b = window
    result = []
    for port in ports:
        l, t, rr, bb = port['box_xyxy']
        if min(rr,r) <= max(l,x) or min(bb,b) <= max(t,y):
            continue
        result.append(dict(class_id=port['class_id'], source_box_index=port['source_box_index'],
            box_xyxy_local=[max(l,x)-x,max(t,y)-y,min(rr,r)-x,min(bb,b)-y],
            complete=x<=l<rr<=r and y<=t<bb<=b))
    return result


def verify_saved_labels(path, labels, window):
    width, height = window[2]-window[0], window[3]-window[1]
    expected = []
    for p in labels:
        l,t,r,b = p['box_xyxy_local']
        expected.append([p['class_id'],l/width,t/height,r/width,t/height,r/width,b/height,l/width,b/height])
    actual = [list(map(float,line.split())) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(actual) == len(expected)
    for a,e in zip(actual,expected):
        assert len(a)==len(e) and all(abs(x-y)<=.000000001 for x,y in zip(a,e))


def main():
    from PIL import Image
    started = time.monotonic()
    if (OUT/'report.json').exists():
        raise FileExistsError('Preserve completed audit')
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    protocol = load(BASE/'protocol.json')
    manifest_path = BASE/'dataset/dataset_manifest.json'
    manifest = load(manifest_path)
    names = sorted(protocol['train_sources'])
    assert len(names) == 192 and set(names).isdisjoint(protocol['inner_val_sources'])
    pins = {str(p): sha(p) for p in (BASE/'protocol.json', manifest_path, OUT/'PLAN.md',
            Path(__file__), Path(__file__).with_name('port_training_multiscale_policy.py'))}
    records = [r for r in manifest['records'] if r['split'] == 'train']
    counts, cases = Counter(), []
    for name in names:
        image_path = DATA/'images/train01'/name
        label_path = DATA/'labels/train01'/(Path(name).stem+'.txt')
        pins[str(image_path)] = sha(image_path)
        pins[str(label_path)] = sha(label_path)
        with Image.open(image_path) as image:
            w, h = image.size
        boxes = parse_boxes(label_path.read_text(encoding='utf-8'), w, h)
        ports = [dict(class_id=p['source_class']-3, source_box_index=i, box_xyxy=p['box_xyxy'])
                 for i, p in enumerate(boxes) if p['source_class'] in (3, 4)]
        planned, skipped = plan_all(ports, [h, w])
        old_exposure, small_exposure, old_sizes = Counter(), Counter(), {}
        for record in records:
            if record['source_image'] != name:
                continue
            # Verify all existing TRAIN examples without reading any heldout content.
            for kind in ('image', 'label'):
                path = BASE/'dataset'/record[kind]
                assert sha(path) == record[kind+'_sha256']
                pins[str(path)] = record[kind+'_sha256']
            if not record['label_count']:
                continue
            window = record.get('window')
            if record['role'] == 'original_replay':
                window = tile_windows(w, h)[record['tile_id']]
            assert window is not None
            labels = (legacy_labels(ports, window) if record['role']=='original_replay'
                      else independent_labels(ports, window, [h,w]))
            assert labels is not None and len(labels) == record['label_count']
            verify_saved_labels(BASE/'dataset'/record['label'], labels, window)
            for label in labels:
                if not label.get('complete',True):
                    counts['original_clipped_label_instances'] += 1
                    continue
                i = label['source_box_index']
                old_exposure[i] += 1
                old_sizes[i] = min(old_sizes.get(i, float('inf')), max(window[2]-window[0], window[3]-window[1]))
        for plan in planned:
            small_exposure.update(p['source_box_index'] for p in plan['labels'])
        counts.update(sources=1, targets=len(ports), safe_windows=len(planned), unsafe_seeds=len(skipped),
                      old_covered=len(old_exposure), new480_covered=len(small_exposure),
                      newly_geometrically_covered=len(set(small_exposure)-set(old_exposure)),
                      covered_at_finer_scale=sum(old_sizes.get(i, float('inf')) > 480 for i in small_exposure),
                      sources_with_480=bool(planned))
        cases.append(dict(source_image=name, shape=[h,w], ports=ports, windows=planned,
                          unsafe_seed_indices=skipped, old_exposure=dict(old_exposure),
                          old_minimum_crop_size=old_sizes, new480_exposure=dict(small_exposure)))
    assert all(sha(Path(p)) == value for p, value in pins.items())
    report = dict(status='complete', counts=dict(counts), pins=pins, cases=cases,
                  original_train_records_verified=len(records), training_started=False,
                  heldout_image_or_label_read=False, production_changed=False,
                  independent_label_geometry_replayed=True,
                  seconds=round(time.monotonic()-started,2),
                  warning='Training geometry only, NOT new detections or field accuracy.')
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('pins','cases')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
