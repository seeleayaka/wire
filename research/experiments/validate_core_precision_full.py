"""Complete original-group inner holdout; locked-rule outer-val verification."""
import argparse, copy, json, os, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REPO = Path('E:/PythonProject10')
sys.path.insert(0, str(REPO))
from core_port_precision_policy import select

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['inner', 'outer'], required=True)
    args = parser.parse_args()
    out = ROOT / ('artifacts/core_precision_full_'+args.split+'_20261002')
    if out.exists():
        raise FileExistsError('Fresh outputs required')
    (out/'config/Ultralytics').mkdir(parents=True)
    os.environ.update(YOLO_CONFIG_DIR=str(out/'config'), YOLO_OFFLINE='True', YOLO_AUTOINSTALL='False')
    import torch
    from core_port_resolution_ab_20261002 import DATA, WEIGHT, MANIFEST, ranking, score
    from inspection_agent.optional_port_crop_review import CONFIG, sha, read_image, predict
    old_env = os.environ['YOLO_CONFIG_DIR']
    # Importing the old scorer sets its own environment; restore isolated runtime.
    os.environ['YOLO_CONFIG_DIR'] = str(out/'config')
    original = copy.deepcopy(CONFIG)
    pilot = ROOT/'artifacts/core_port_resolution_ab_20261002'
    pilot_protocol = load(pilot/'protocol.json')
    assert sha(WEIGHT) == pilot_protocol['weight_sha256'] and CONFIG == pilot_protocol['baseline_config']
    manifest = load(MANIFEST)
    train = {r['source_image'] for r in manifest['records'] if r['split']=='train'}
    inner = {r['source_image'] for r in manifest['records'] if r['split']=='val'}
    assert train.isdisjoint(inner)
    modes = ['bounded', 'high_score', 'weak_overlap_consensus']
    selected = None
    if args.split == 'inner':
        names = sorted(inner)
        source_split = 'train01'
    else:
        decision = load(ROOT/'artifacts/core_precision_full_inner_20261002/report.json')
        selected = decision['selected_policy']
        modes = ['bounded'] + ([selected] if selected != 'bounded' else [])
        source_split = 'val01'
        names = sorted(p.name for p in (DATA/'images/val01').glob('*.JPG'))
    assert len(names) == (48 if args.split=='inner' else 30)
    paths = [DATA/'images'/source_split/n for n in names]
    protected = [WEIGHT, MANIFEST, REPO/'inspection_agent/optional_port_crop_review.py',
                 REPO/'inspection_agent/port_tiling.py', ROOT/'experiments/core_port_precision_policy.py']
    hashes = {str(p):sha(p) for p in protected+paths}
    save(out/'protocol.json', dict(split=source_split, names=names, modes=modes, selected_policy=selected,
        hashes=hashes, config=original, score='same-class IoU>=0.5, one-to-one',
        selection_gate='TP >= bounded TP AND unmatched < bounded unmatched; choose lowest unmatched then highest TP',
        extra_threshold=.5, weak_first_threshold=.25, maximum=5, consensus_support_threshold=.25,
        consensus_iou=.5, consensus_margin=16, labels_used_for_inference=False,
        inner_previous_model_selection=True, outer_previously_inspected=True, training=False))
    from ultralytics import YOLO
    torch.set_num_threads(4)
    torch.manual_seed(20260929)
    model = YOLO(str(WEIGHT))
    assert model.task == 'segment' and dict(model.names) == {0:'unplugged_plug', 1:'unplugged_jack'}
    cases = []
    tick = time.monotonic()
    for path in paths:
        cached = pilot/(path.stem+'_predictions.json')
        reuse = args.split=='inner' and path.name in pilot_protocol['images']
        if reuse:
            assert sha(path) == pilot_protocol['hashes'][str(path)]
            raw = load(cached)['predictions']['960']
        else:
            raw = predict(model, read_image(path))
        case = dict(image=path.name, predictions=raw, cached=reuse,
                    source_sha256=sha(path), weight_sha256=sha(WEIGHT))
        save(out/(path.stem+'_predictions.json'), case)
        cases.append(case)
        save(out/'progress.json', dict(completed=len(cases), total=len(paths), elapsed_seconds=round(time.monotonic()-tick,2)))
        print(f'{len(cases)}/{len(paths)} {path.name} cache={reuse}', flush=True)
    # Do not open any labels until all selections/inference have completed.
    chosen = {c['image']:{m:select(c['predictions'],m) for m in modes} for c in cases}
    summary = {}
    groups = {}
    for case in cases:
        label = DATA/'labels'/source_split/(Path(case['image']).stem+'.txt')
        targets = []
        height, width = case['predictions']['source_shape']
        for line in label.read_text(encoding='utf-8').splitlines():
            cls, cx, cy, w, h = map(float, line.split())
            if cls in (3,4):
                targets.append(dict(class_id=int(cls)-3, box=[(cx-w/2)*width,(cy-h/2)*height,(cx+w/2)*width,(cy+h/2)*height]))
        result = {}
        stages = {'top1':ranking(case['predictions']['merged_predictions'],1), **chosen[case['image']]}
        group = case['image'].split('_')[0]
        for mode, rows in stages.items():
            metrics = score(rows, targets)
            result[mode] = metrics
            for aggregate in (summary.setdefault(mode,{k:0 for k in metrics}),
                              groups.setdefault(group,{}).setdefault(mode,{k:0 for k in metrics})):
                for k,v in metrics.items(): aggregate[k] += v
        case['metrics'] = result
        case['label_sha256'] = sha(label)
        save(out/(Path(case['image']).stem+'_evaluation.json'),dict(metrics=result, selected=chosen[case['image']]))
    base = summary['bounded']
    if args.split == 'inner':
        qualified = [m for m in modes if m!='bounded' and summary[m]['tp']>=base['tp'] and summary[m]['unmatched']<base['unmatched']]
        selected = min(qualified, key=lambda m:(summary[m]['unmatched'],-summary[m]['tp'],m)) if qualified else 'bounded'
    assert CONFIG == original and {p:sha(Path(p)) for p in hashes} == hashes
    save(out/'report.json',dict(status='complete', summary=summary, groups=groups, selected_policy=selected,
        cases=[{k:c[k] for k in ('image','cached','metrics','source_sha256','label_sha256')} for c in cases],
        new_inferences=sum(not c['cached'] for c in cases), cached_inferences=sum(c['cached'] for c in cases),
        seconds=round(time.monotonic()-tick,2), formal_policy_changed=False,
        accuracy_scope='retrospective same-camera port localization only; no electrical verdict or unseen cabinet claim'))
    print(json.dumps(dict(summary=summary,selected=selected),indent=2),flush=True)

if __name__=='__main__':main()
