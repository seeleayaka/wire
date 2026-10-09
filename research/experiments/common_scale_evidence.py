"""Four actual view recipes, still exactly three checkpoint identities.

Unlaunched common640 experiment preparation. No model calls or GT access.
Keep the completed/current three-view auditor immutable.
"""
import math
from audit_allport480_source import audit_views
from inspection_agent.optional_port_crop_review import CONFIG
from inspection_agent.port_tiling import tile_windows, near_artificial_edge, merge_tiled_ports
from replacement_voters import check_prefix


def audit_common_views(evidence, roles, source_sha, shape):
    height, width = shape
    grid = [list(w) for w in tile_windows(width, height, 640, 480)]
    extra = [v for v in evidence if v['predictions'].get('windows') == grid]
    previous = [v for v in evidence if v['predictions'].get('windows') != grid]
    if len(extra) != 3 or {v['weight_sha256'] for v in extra} != set(roles.values()):
        raise ValueError('exactly one actual common640 view per checkpoint required')
    audit_views(previous, roles, source_sha, shape)
    for view in extra:
        pred = view['predictions']
        if view['source_sha256'] != source_sha or pred['source_shape'] != list(shape):
            raise ValueError('common640 source/frame identity drift')
        raw, rejected = pred.get('raw_predictions'), pred.get('edge_rejected')
        kept = pred.get('edge_kept_predictions')
        if (type(raw) is not int or type(rejected) is not int or not isinstance(kept, list)
                or rejected < 0 or raw < rejected or raw != rejected + len(kept)):
            raise ValueError('actual tile count conservation failed')
        for row in kept:
            box, tile, cls, confidence = row['box_xyxy'], row['source_tile'], row['class_id'], row['confidence']
            if type(tile) is not int or not 0 <= tile < len(grid):
                raise ValueError('invalid original tile identity')
            if (len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                    or type(cls) is not int or cls not in (0, 1)
                    or type(confidence) not in (int, float) or not math.isfinite(confidence)
                    or not 0 <= confidence <= 1):
                raise ValueError('invalid actual common640 detection')
            x, y, right, bottom = grid[tile]
            left, top, r, b = box
            if not x <= left < r <= right or not y <= top < b <= bottom:
                raise ValueError('detection outside the tile that generated it')
            local = [left-x, top-y, r-x, b-y]
            if near_artificial_edge(local, grid[tile], width, height, CONFIG['edge_margin']):
                raise ValueError('artificial-cut detection cannot become kept evidence')
        if merge_tiled_ports(kept, CONFIG['cross_tile_nms_iou']) != pred['merged_predictions']:
            raise ValueError('common640 merge replay mismatch')
    return 12


def validate_fallback_prerequisites(source, source_audit, development, development_audit):
    """A source PASS is not deployment; finish the current validation first."""
    if (source.get('qualifies') is not True or source_audit.get('status') != 'pass'
            or source_audit.get('candidate_source_qualifies') is not True
            or source.get('summary', {}).get('trial', {}).get('tp') != 300
            or source.get('summary', {}).get('trial', {}).get('unmatched') != 4
            or source.get('normal_cues') != 0):
        raise ValueError('completed audited teacher SOURCE300/4 prerequisite required')
    # The separate runner must additionally bind report/protocol/case file SHAs,
    # accepted prefixes, exact cohort, mainline fingerprint and dirty worktree.
    if (development.get('status') != 'rejected' or development.get('failed_stage') not in ('inner', 'outer')
            or development_audit.get('status') != 'pass'
            or development_audit.get('candidate_development_qualifies') is not False):
        raise ValueError('only a complete independently replayed development rejection permits fallback')
    failed = development['failed_stage']
    stages = development.get('stages', {})
    expected_order = ['inner'] if failed == 'inner' else ['inner', 'outer']
    if (list(stages) != expected_order or stages[failed].get('qualifies') is not False
            or stages[failed].get('count') != (48 if failed == 'inner' else 30)
            or (failed == 'outer' and stages['inner'].get('qualifies') is not True)
            or stages != development_audit.get('stages')):
        raise ValueError('final rejection stage population/audit replay mismatch')
    return True


def check_common_prefixes(accepted, research, teacher, trial):
    for older, newer in [(accepted, research), (research, teacher), (teacher, trial)]:
        check_prefix(older, newer, newer)
    return True


def common_source_qualifies(totals, normal, rows):
    """Strict net gain over300, not a repeat of the prior298 source threshold."""
    for name, tp in [('original', 295), ('research', 298), ('current', 300)]:
        if totals.get(name) != dict(tp=tp, unmatched=4, fn=344-tp,
                                   predictions=tp+4, targets=344):
            raise ValueError('wrong protected source generation: ' + name)
    if len(rows) != 192 or len({r['image'] for r in rows}) != 192:
        raise ValueError('complete unique TRAIN192 required before source acceptance')
    trial = totals['trial']
    if (trial['targets'] != 344 or trial['fn'] != 344-trial['tp']
            or trial['predictions'] != trial['tp']+trial['unmatched']):
        raise ValueError('invalid full-source metrics')
    return (trial['tp'] > 300 and trial['unmatched'] <= 4 and normal == 0
            and not any(r['lost'] or r['lost_original'] or r['lost_research'] for r in rows))
