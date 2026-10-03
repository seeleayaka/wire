"""Summarize a completed A/B run, including review burden, not field accuracy."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'artifacts/rescue_mainline_ab_20261002_v2'
def main():
    report=json.loads((ROOT/'report.json').read_text(encoding='utf-8'))
    assert report['status']=='complete'
    sums={stage:{key:sum(row[stage][key] for row in report['cases']) for key in (
        'candidate_count','target_count','target_any_overlap','target_iou050','candidate_iou050')}
        for stage in ('off','on','added')}
    summary=dict(stages=sums, precise_target_gain=sums['on']['target_iou050']-sums['off']['target_iou050'],
        extra_review_boxes=sums['added']['candidate_count'],
        added_unmatched_at_iou050=sums['added']['candidate_count']-sums['added']['candidate_iou050'],
        normal_baseline_review_boxes=sum(row['off']['candidate_count'] for row in report['cases'] if row['image'].startswith('normal')),
        normal_added_boxes=sum(row['new_hints'] for row in report['cases'] if row['image'].startswith('normal')),
        default_enabled=False,decision='retain optional bounded supplement, not baseline replacement',
        mainline_new_initial_reports=True,sam_raw_masks_reused=True,field_accuracy_claimed=False,
        sample_limit='four previously inspected images from same camera/dataset; not an unseen validation set')
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
