"""Read-only class/geometry audit of failed fine-head feasibility, not retuning."""
import json
import sys
from collections import Counter
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, 'E:/PythonProject10')
from inspection_agent.optional_port_crop_review import sha
from audit_port_multiscale_acceptance import matches, metric, overlap


def main():
    import torch
    source = ROOT/'artifacts/dense_pixel_port_probe_20261003/full'
    out = ROOT/'artifacts/dense_pixel_error_audit_20261003'
    if out.exists():
        raise FileExistsError('Preserve prior audit')
    report = json.loads((source/'report.json').read_text(encoding='utf-8'))
    saved = json.loads((source/'inner_crop_evaluation.json').read_text(encoding='utf-8'))
    assert report['status'] == 'complete' and saved['summary'] == report['crop_feasibility']
    assert sha(Path(report['checkpoint']['path'])) == report['checkpoint']['sha256']
    out.mkdir()
    reasons = Counter()
    class_scores = {c:Counter() for c in (0,1)}
    rows = []
    totals = Counter()
    for case in saved['cases']:
        path = source/'features/inner_val'/f"{case['crop_index']:04d}.pt"
        assert sha(path) == report['cache_pins'][str(path)]
        item = torch.load(path, map_location='cpu', weights_only=True)
        preds, targets = case['predictions'], item['boxes']
        assert item['source_image'] == case['source_image']
        scored = metric(preds, targets)
        assert scored == case['metrics']
        totals.update(scored)
        used, extra, pairs = matches(preds, targets)
        for c in (0,1):
            class_scores[c].update(metric([p for p in preds if p['class_id']==c], [t for t in targets if t['class_id']==c]))
        for pi in sorted(extra):
            p = preds[pi]
            same = [(overlap(p['box_xyxy'],t['box']),ti) for ti,t in enumerate(targets) if t['class_id']==p['class_id']]
            opposite = [(overlap(p['box_xyxy'],t['box']),ti) for ti,t in enumerate(targets) if t['class_id']!=p['class_id']]
            value, ti = max(same, default=(0,None))
            other, oi = max(opposite, default=(0,None))
            if value>=.5:
                reason='duplicate'
            elif other>=.5:
                reason='class_mismatch'
            elif value>=.1:
                reason='localization_overlap_below_half'
            else:
                reason='no_same_class_overlap'
            reasons[reason]+=1
            rows.append(dict(crop_index=case['crop_index'],source_image=case['source_image'],reason=reason,
                prediction=p,best_same_iou=value,best_same_target=targets[ti] if ti is not None else None,
                best_other_iou=other,best_other_target=targets[oi] if oi is not None else None))
    assert dict(totals)==saved['summary']['metrics']
    result=dict(status='complete',coarse_precision=.7032258064516129,fine_summary=saved['summary'],
        error_reasons=dict(reasons),per_class={str(c):dict(v) for c,v in class_scores.items()},
        unmatched=rows,checkpoint_sha256=report['checkpoint']['sha256'],
        fixed_score=.5,no_threshold_search=True,source_accuracy=False,production_changed=False)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='unmatched'},indent=2))


if __name__=='__main__':
    main()
