"""Construct proposals before TRAIN GT audit; ceiling is not accuracy."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from prepare_paired_semantic_geometry import NEW as GEOMETRY
from current_port_baseline_audit import BASE, read_targets, read_current_case
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint
from inspection_agent.port_tiling import box_iou
from audit_port_matching_cardinality import maximum_matching
from paired_median_proposals import proposals
OUT = ROOT / 'artifacts/paired_median_train_ceiling_20261003'
PREP = ROOT / 'artifacts/paired_support_graph_20261003/source_selections/train/index.json'
EXTENDED = ROOT / 'artifacts/paired_multimodel_proposal_ceiling_20261003'


def main():
    if OUT.exists(): raise FileExistsError('Preserve TRAIN-only geometry audit')
    frozen = paired_runtime_fingerprint(REPO)
    pins = {str(p): sha(p) for p in (Path(__file__), Path(__file__).with_name('paired_median_proposals.py'),
        ROOT / 'artifacts/paired_median_geometry_preregistration_20261003/PLAN.md', PREP, BASE / 'train/report.json')}
    OUT.mkdir(); records = []; items = load(PREP)['records']; assert len(items) == 192
    entries = {r['image']: r for r in load(BASE / 'train/report.json')['cases']}
    for item in items:
        path = Path(item['path']); assert sha(path) == item['sha256']; pins[str(path)] = sha(path)
        case = load(path); name = item['image']; entry = entries[name]
        teacher, baseline = read_current_case('train', entry, pins); assert case['teacher'] == teacher
        accepted_path = GEOMETRY / 'full_train' / (Path(name).stem + '_predictions.json')
        pins[str(accepted_path)] = sha(accepted_path); current = load(accepted_path)['trial']; old = current['all_predictions']
        remaining = 5 - (len(old) - len(current['primary'])); assert remaining >= 0
        candidates = proposals(teacher, [teacher, case['student'], case['feature'], baseline['alternative']]) if remaining else []
        candidates = [p for p in candidates if not any(box_iou(p['box_xyxy'], q['box_xyxy']) >= .5 for q in old)]
        extended_path = EXTENDED / ('train_' + Path(name).stem + '_proposals.json'); pins[str(extended_path)] = sha(extended_path)
        previous = load(extended_path); assert previous['current'] == current
        original = previous['extended_candidates']
        save(OUT / (Path(name).stem + '_proposals.json'), dict(image=name, current=current, candidates=candidates,
            original_candidates=original, remaining=remaining, GT_not_used=True))
        targets = read_targets('train', name, teacher['predictions']['source_shape'], entry['label_sha256'], pins)
        matched = maximum_matching(old, targets); current_tp = len(matched)
        upper = lambda rows: min(len(maximum_matching(old + rows, targets)), current_tp + remaining)
        missing = []
        raw = [r for m in (teacher, case['student'], case['feature'], baseline['alternative']) for r in m['predictions']['merged_predictions'] if r['confidence'] > .05]
        for ti, target in enumerate(targets):
            if ti in matched: continue
            best = lambda rows: max([box_iou(r['box_xyxy'], target['box']) for r in rows if r['class_id'] == target['class_id']], default=0.)
            missing.append(dict(target=ti, class_id=target['class_id'], original_best_iou=best(original), median_best_iou=best(candidates),
                raw_single_checkpoint_best_iou=best(raw), no_raw_overlap50=best(raw) < .5, budget_full=remaining == 0))
        records.append(dict(image=name, current_tp=current_tp, targets=len(targets), remaining=remaining,
            original_upper=upper(original), median_upper=upper(candidates), union_upper=upper(original + candidates),
            original_candidates=len(original), median_candidates=len(candidates), missing=missing))
    totals = {key: sum(r[key] for r in records) for key in ('current_tp', 'targets', 'original_upper', 'median_upper', 'union_upper', 'original_candidates', 'median_candidates')}
    assert totals['current_tp'] == 286 and totals['targets'] == 344
    missing = [m for r in records for m in r['missing']]
    funnel = dict(missed_targets=len(missing), no_raw_single_checkpoint_iou50=sum(r['no_raw_overlap50'] for r in missing),
        budget_full=sum(r['budget_full'] for r in missing), original_candidate_iou50=sum(r['original_best_iou'] >= .5 for r in missing),
        median_candidate_iou50=sum(r['median_best_iou'] >= .5 for r in missing),
        newly_localized_by_median=sum(r['median_best_iou'] >= .5 > r['original_best_iou'] for r in missing))
    assert {p: sha(Path(p)) for p in pins} == pins and paired_runtime_fingerprint(REPO) == frozen
    save(OUT / 'report.json', dict(status='complete', summary=totals, funnel=funnel, cases=records, pins=pins,
        TRAIN_only=True, GT_after_proposals=True, ceiling_not_accuracy=True, no_model_inference=True, no_deployment=True, field_accuracy=False))
    print(str(dict(summary=totals, funnel=funnel)), flush=True)


if __name__ == '__main__': main()
