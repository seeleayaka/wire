"""One preset stability veto on all saved TRAIN poses, no new inference."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, load, save, sha
from current_port_baseline_audit import BASE, read_targets
from audit_port_multiscale_acceptance import metric, matches
from paired_pose_consistency import select
from inspection_agent.paired_port_geometry import HEAD_SHA
SOURCE = ROOT / 'artifacts/paired_pose_search_20261003'
OUT = ROOT / 'artifacts/paired_pose_consistency_20261003'


def main():
    if OUT.exists(): raise FileExistsError('Preserve fixed stability control')
    OUT.mkdir(); pins = {str(p):sha(p) for p in (Path(__file__), Path(__file__).with_name('paired_pose_consistency.py'))}
    source_report = load(SOURCE / 'report.json'); assert source_report['status'] == 'rejected'
    pins.update(source_report['pins']); assert {p:sha(Path(p)) for p in pins} == pins
    rows = []
    for entry in load(BASE / 'train/report.json')['cases']:
        name = entry['image']; path = SOURCE / 'train' / (Path(name).stem + '_predictions.json')
        pins[str(path)] = sha(path); record = load(path)
        current = record['current']; trial = select(current, record['proposals'], record['probabilities'], HEAD_SHA)
        save(OUT / path.name, dict(image=name,current=current,trial=trial))
        targets = read_targets('train', name, [2736,3648], entry['label_sha256'], pins)
        oh, nh = matches(current['all_predictions'], targets)[0], matches(trial['all_predictions'], targets)[0]
        rows.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),lost=sorted(oh-nh),gained=sorted(nh-oh)))
    totals = {v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
    assert totals['current']['tp'] == 289
    qualifies = totals['trial']['tp'] > 289 and totals['trial']['unmatched'] <= 4 and not any(r['lost'] for r in rows)
    assert {p:sha(Path(p)) for p in pins} == pins
    result = dict(status='complete',qualifies=qualifies,summary=totals,cases=rows,pins=pins,
        same_raw_probability_gate=True,no_new_inference=True,no_deployment=True,field_accuracy=False)
    save(OUT / 'report.json',result); print(str(dict(qualifies=qualifies,summary=totals)),flush=True)


if __name__ == '__main__': main()
