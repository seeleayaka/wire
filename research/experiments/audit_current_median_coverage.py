"""Post-selection raw detector coverage diagnosis; never feed GT into policy."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from current_port_baseline_audit import BASE,read_current_case,read_targets
from audit_port_multiscale_acceptance import matches,metric,overlap
from paired_median_proposals import proposals
from inspection_agent.paired_median_geometry import median_runtime_fingerprint
MODELS = ROOT / 'artifacts/paired_support_graph_20261003/source_selections'
MEDIAN = ROOT / 'artifacts/paired_median_current_head_20261003'
OUT = ROOT / 'artifacts/current_median_raw_coverage_audit_20261003'


def main():
    if OUT.exists(): raise FileExistsError('Preserve raw coverage audit')
    pins = {str(Path(__file__)):sha(Path(__file__))}; frozen = median_runtime_fingerprint(REPO)
    records = []; summary = {}
    for stage,count in (('train',192),('inner',48),('outer',30)):
        index_path = MODELS / stage / 'index.json'; pins[str(index_path)] = sha(index_path)
        indexed = {r['image']:r for r in load(index_path)['records']}; entries = load(BASE / stage / 'report.json')['cases']; assert len(entries) == count
        counters = {}; unavoidable = 0; misses = 0
        for entry in entries:
            name = entry['image']; path = Path(indexed[name]['path']); assert sha(path) == indexed[name]['sha256']; pins[str(path)] = sha(path)
            case = load(path); teacher,old = read_current_case(stage,entry,pins); assert teacher == case['teacher']
            modelset = [teacher,case['student'],case['feature'],old['alternative']]
            fixed = MEDIAN / stage / (Path(name).stem+'_predictions.json'); pins[str(fixed)] = sha(fixed); saved = load(fixed)
            current = saved['trial']; raw = [(m['weight_sha256'],p) for m in modelset for p in m['predictions']['merged_predictions']]
            generic = proposals(teacher,modelset)
            # All runtime geometry and original predictions are fixed BEFORE labels.
            targets = read_targets(stage,name,[2736,3648],entry['label_sha256'],pins)
            matched = matches(current['all_predictions'],targets)[0]; room = 5-(len(current['all_predictions'])-len(current['primary']))
            unavoidable += max(0,len(targets)-len(current['primary'])-5)
            for ti in sorted(set(range(len(targets)))-matched):
                target = targets[ti]; cls = target['class_id']
                geometries = [(weight,p) for weight,p in raw if overlap(p['box_xyxy'],target['box']) >= .5]
                same = [(weight,p) for weight,p in geometries if p['class_id'] == cls]
                weak = [(weight,p) for weight,p in same if p['confidence'] > .05]
                h,w = teacher['predictions']['source_shape']
                complete = [(weight,p) for weight,p in weak if 16 <= p['box_xyxy'][0] < p['box_xyxy'][2] <= w-16 and 16 <= p['box_xyxy'][1] < p['box_xyxy'][3] <= h-16]
                voted = [p for p in generic if p['class_id'] == cls and overlap(p['box_xyxy'],target['box']) >= .5]
                evaluated = [(p,s) for p,s in zip(saved['proposals'],saved['probabilities']) if p['class_id'] == cls and overlap(p['box_xyxy'],target['box']) >= .5]
                reason = ('shared_extra_budget_full' if not room else 'no_precise_box_any_class' if not geometries
                    else 'precise_box_wrong_class_only' if not same else 'sameclass_only_below_weak_floor' if not weak
                    else 'weak_box_incomplete_frame_margin' if not complete else 'single_checkpoint_precise_only' if len({weight for weight,p in complete}) < 2
                    else 'median_geometry_or_suppression' if not voted else 'overlap_or_valid_context_suppression' if not evaluated
                    else 'frozen_semantic_or_selection_gate')
                counters[reason] = counters.get(reason,0)+1; misses += 1
                records.append(dict(stage=stage,image=name,target_index=ti,class_id=cls,reason=reason,
                    precise_checkpoints=len({weight for weight,p in complete}),any_class_boxes=len(geometries),sameclass_boxes=len(same),
                    generic_median_boxes=len(voted),evaluated_boxes=len(evaluated),
                    best_raw_class_probability=max((s[cls+1] for p,s in evaluated),default=None)))
        assert misses == load(MEDIAN / stage / 'report.json')['summary']['trial']['fn']
        summary[stage] = dict(misses=misses,causes=counters,unavoidable_fixed_budget_lower_bound=unavoidable)
    assert {p:sha(Path(p)) for p in pins} == pins and median_runtime_fingerprint(REPO) == frozen
    OUT.mkdir(); save(OUT / 'report.json',dict(status='complete',summary=summary,cases=records,pins=pins,
        GT_diagnostic_only=True,no_proposal_rules_changed=True,no_deployment=True,field_accuracy=False))
    print(str(summary),flush=True)


if __name__ == '__main__': main()
