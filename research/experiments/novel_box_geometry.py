"""Class-agnostic duplicate protection for new cues, no GT or image IDs."""
from relative_port_box import geometry


def intersection_over_smaller(a,b):
    _,_,aw,ah=geometry(a);_,_,bw,bh=geometry(b)
    area=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    return area/min(aw*ah,bw*bh)


def is_duplicate(box,rows):
    return any(intersection_over_smaller(box,r['box_xyxy'])>=.5 for r in rows)


def select(current,candidates,probabilities,head_sha):
    from inspection_agent.paired_native_pose_features import select as original_select
    # Keep the original chooser/probability validation and fixed parent ordering.
    # Filter once before choosing; a competing nested candidate must not consume
    # its parent's selection slot before a distinct alternative is considered.
    original_select(current,candidates,probabilities,head_sha)
    eligible=[i for i,r in enumerate(candidates) if not is_duplicate(r['box_xyxy'],current['all_predictions'])]
    chosen=original_select(current,[candidates[i] for i in eligible],[probabilities[i] for i in eligible],head_sha)
    # Exclusion-only refill, preserving original ordering and head scores.
    # Re-run finite original selection if a newly chosen pair duplicates, rather
    # than editing prefix or reordering by source/GT.
    excluded=set();limit=len(eligible)+1
    for _ in range(limit):
        kept=[];bad=None
        for row in chosen['paired_semantic_additions']:
            if is_duplicate(row['box_xyxy'],kept):bad=row;break
            kept.append(row)
        if bad is None:return chosen
        key=(bad['class_id'],*bad['box_xyxy']);excluded.add(key)
        indices=[i for i in eligible if (candidates[i]['class_id'],*candidates[i]['box_xyxy']) not in excluded]
        chosen=original_select(current,[candidates[i] for i in indices],[probabilities[i] for i in indices],head_sha)
    raise AssertionError('Finite duplicate exclusion did not converge')
