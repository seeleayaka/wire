"""Diagnostic replay, not approved topology integration or fresh SAM inference."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments' / 'route_topology'))
sys.path.insert(0, 'E:/PythonProject10')
from core import extract_mask, assess_view, compare_views
from inspection_agent.workflow import InspectionTask


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = ROOT / 'artifacts/local_performance_fresh_20261007/cabinet_4_wrong3'
    visual_path = base / 'desktop_output/20261007_204534/report.json'
    visual = json.loads(visual_path.read_text(encoding='utf-8'))
    out = ROOT / 'artifacts/cabinet4_topology_box_probe_20261008'
    out.mkdir(exist_ok=False)
    sources = {
        'reference': base / 'fresh_sam_cache/reference_masks/57f4eb2143f2c0e84d43',
        'inspection': visual_path.parent / 'sam3/inspection',
    }
    pinned = {visual_path: digest(visual_path),
              ROOT / 'experiments/route_topology/core.py': digest(ROOT / 'experiments/route_topology/core.py')}
    views, summaries = {}, {}
    for side, directory in sources.items():
        report_path = directory / 'report.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        paths = sorted(p for p in directory.glob('mask_*.png') if p.stem[5:].isdigit())
        assert len(paths) == report['instance_count'] == len(report['scores'])
        assert [p.name for p in paths] == [f'mask_{i:03}.png' for i in range(1, len(paths) + 1)]
        pinned[report_path] = digest(report_path)
        pinned[directory / 'input.jpg'] = digest(directory / 'input.jpg')
        with Image.open(directory / 'input.jpg') as img:
            size = list(img.size)
        records, overlaps = [], {r['id']: [] for r in visual['review_regions']}
        for i, path in enumerate(paths):
            pinned[path] = digest(path)
            with Image.open(path) as image:
                mask = np.asarray(image.convert('L')) > 0
            assert list(mask.shape[::-1]) == size
            record = extract_mask(mask, report['scores'][i], path.stem)
            records.append(record)
            for region in visual['review_regions']:
                x1, y1, x2, y2 = region['bbox_xyxy']
                count = int(mask[y1:y2, x1:x2].sum())
                if count:
                    overlaps[region['id']].append({'mask_id': path.stem, 'pixels_inside_box': count,
                                                   'score': record['score'], 'geometry_state': record['geometry']['state']})
        # No fabricated terminal identities, scope confirmation, expected wiring,
        # or handpicked convenient mask selection for this cabinet.
        views[side] = assess_view(records, [], size, {'confirmed': False})
        summaries[side] = {'mask_count': len(records),
                           'states': dict(Counter(r['state'] for r in views[side]['evidence'])),
                           'regions': overlaps}
        print(json.dumps({'side': side, **{k: v for k, v in summaries[side].items() if k != 'regions'}}, ensure_ascii=False), flush=True)
    topology = compare_views(views['reference'], views['inspection'], None, {'confirmed': False})
    # Exercise actual Agent import. Experimental core returns an independent
    # topology verdict, not a replacement review_regions array.
    task = InspectionTask('cabinet4-topology-box-replay', 'cabinet', str(visual['reference']), str(visual['inspection']))
    task.record_visual_analysis(visual, source_report=visual_path)
    imported = task.to_report()['machine_evidence'][0]
    before = [r['id'] for r in visual['review_regions']]
    after = imported['candidate_ids']
    assert before == after
    assert topology['decision'] == 'insufficient_evidence'
    assert 'green_02' in after
    assert all(digest(p) == d for p, d in pinned.items())
    result = {
        'mode': 'existing_actual_sam_masks_new_topology_analysis',
        'fresh_sam_inference': False, 'production_modified': False,
        'accepted_source_pixel_route_review': False,
        'provenance_limit': 'Historical cabinet masks include registered inspection pixels; not new approved source-pixel manifests.',
        'confirmed_terminal_map_available': False, 'expected_wiring_available': False,
        'before_visual_ids': before, 'after_agent_import_ids': after,
        'bottom_box_retained': 'green_02' in after,
        'topology_can_clear_bottom_box': False,
        'topology_decision': topology['decision'], 'topology_reasons': topology['reasons'],
        'mask_summaries': summaries,
        'actual_agent_visual_evidence': imported,
        'source_fingerprints': [{'path': str(p), 'sha256': d} for p, d in pinned.items()],
        'not_claimed': ['electrical continuity', 'confirmed false positive', 'full topology pipeline passed', 'field accuracy'],
    }
    (out / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'route_analysis.json').write_text(json.dumps(topology, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('bottom_box_retained', 'topology_decision', 'before_visual_ids', 'after_agent_import_ids')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
