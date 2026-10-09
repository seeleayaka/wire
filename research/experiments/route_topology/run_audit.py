"""Constructed end-to-end checks + fixed historical real-mask geometry audit.

No model inference, accuracy score, image-name tuning or genuine port labels.
"""
import json
import argparse
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

from core import image_binding, sha256, extract_mask, assess_view
from run_review import verified_run, execute, save
from test_pipeline import create_fixture
from test_core import mask, REVIEW


ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path('E:/PythonProject10')
OUT = ROOT / 'artifacts/route_topology_20261005'


def source_pins():
    files = ['inspection_agent/topology.py', 'inspection_agent/terminal_mapping.py',
             'inspection_agent/visible_segment_geometry.py', 'inspection_agent/sam_topology_adapter.py',
             'prototype/sam3_wire_fusion.py', 'inspection_agent/workflow.py',
             'prototype/assembly_auto_review_dino.py']
    pins = {str(PROJECT / p): sha256(PROJECT / p) for p in files}
    status = subprocess.check_output(['E:/Git/cmd/git.exe', '-C', str(PROJECT), 'status', '--porcelain'], text=True)
    import hashlib
    pins['dirty_git_status_sha256'] = hashlib.sha256(status.encode()).hexdigest()
    return pins


def diagram():
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    sheet = Image.new('RGB', (1200, 430), '#f7f8fa')
    d = ImageDraw.Draw(sheet)
    d.text((22, 12), '软件构造测试：不是实拍识别成功或准确率', fill='#293645', font=font)
    cases = [('同端子、换路线', [(20, 30), (20, 60), (100, 60), (100, 30)], '同一可见端子关系', '#177750'),
             ('换到另一端子', [(20, 30), (20, 80), (100, 80)], '可见关系变化 A→C', '#ac482b'),
             ('中间遮挡 / 缺段', [(20, 30), (100, 30)], '证据不足，不猜连接', '#806024')]
    for i, (title, points, verdict, color) in enumerate(cases):
        x = 20 + i * 395
        d.rectangle((x, 60, x + 375, 410), fill='white', outline='#dce1e6', width=1)
        d.text((x + 15, 74), title, fill='#263342', font=font)
        d.text((x + 15, 365), verdict, fill=color, font=small)
        transform = lambda p: (x + 20 + p[0] * 2.7, 120 + p[1] * 2.5)
        d.line([transform((20, 30)), transform((100, 30))], fill='#cdd3da', width=3)
        if i == 2:
            d.line([transform((20, 30)), transform((50, 30))], fill='#2587c4', width=5)
            d.line([transform((70, 30)), transform((100, 30))], fill='#2587c4', width=5)
            d.rectangle((*transform((50, 22)), *transform((70, 38))), fill='#e3e5e9')
        else:
            d.line([transform(p) for p in points], fill='#2587c4', width=5)
        for identity, p in [('A', (20, 30)), ('B', (100, 30)), ('C', (100, 80))]:
            px, py = transform(p)
            d.ellipse((px - 7, py - 7, px + 7, py + 7), fill='#fff', outline='#1764b1', width=2)
            d.text((px - 7, py - 34), identity, font=small, fill='#263342')
    sheet.save(OUT / 'constructed_cases.png')


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    OUT = parser.parse_args().output.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    pins = source_pins()
    save(OUT / 'preregistration.json', {'baseline_pins': pins, 'minimum_score': .75,
         'fixed_runs': ['bound_sam_topology_20261001/cabinet_1_fresh',
                       'sam_crop_coverage_20261001/cabinet_1_crop_run',
                       'sam_crop_coverage_20261001/cabinet_2_crop_run'],
         'no_model_inference': True, 'no_real_port_confirmation': True,
         'no_scene_tuning': True, 'no_gt_change': True})
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern='test_*.py')
    with (OUT / 'tests.log').open('w', encoding='utf-8') as stream:
        test = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    save(OUT / 'tests.json', {'tests': test.testsRun, 'success': test.wasSuccessful(),
                             'failures': len(test.failures), 'errors': len(test.errors)})
    if not test.wasSuccessful():
        raise RuntimeError('regression tests failed')
    ref_run, ref_side = create_fixture(OUT, 'constructed_ref', mask([(20, 30), (100, 30)]))
    cases = {'route_change': mask([(20, 30), (20, 60), (100, 60), (100, 30)]),
             'changed_terminal': mask([(20, 30), (20, 80), (100, 80)]),
             'occlusion': mask([(20, 30), (100, 30)])}
    cases['occlusion'][:, 50:70] = 0
    results = {}
    for name, pixels in cases.items():
        run, side = create_fixture(OUT, 'constructed_' + name, pixels)
        pair = {'schema_version': 1, 'scene_type': 'constructed_software_fixture',
                'reference': ref_side, 'inspection': side,
                'expected_connections': [{'from': 'A', 'to': 'B'}], 'expected_review': REVIEW}
        pair_path = OUT / (name + '_pair.json')
        save(pair_path, pair)
        results[name] = execute(pair_path, ref_run, run, OUT / (name + '_review'))['decision']
    diagram()
    sys.path.insert(0, str(PROJECT))
    from inspection_agent.visible_segment_geometry import assess_visible_skeleton
    import cv2
    diagnostics = []
    for relative in ['bound_sam_topology_20261001/cabinet_1_fresh',
                     'sam_crop_coverage_20261001/cabinet_1_crop_run',
                     'sam_crop_coverage_20261001/cabinet_2_crop_run']:
        run = ROOT / 'artifacts' / relative
        manifest = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8'))
        source = Path(manifest['image_binding']['image_path'])
        origin = verified_run(run, source)
        records, comparisons = [], []
        for i, path in enumerate(origin['paths']):
            with Image.open(path) as opened:
                a = np.asarray(opened.convert('L'))
            records.append(extract_mask(a, origin['scores'][i], path.stem))
            # Identical original components >=50px, for diagnostic comparability.
            count, labels, stats, _ = cv2.connectedComponentsWithStats((a > 0).astype(np.uint8), connectivity=8)
            for component in range(1, count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                if area < 50:
                    continue
                skeleton = skeletonize(labels == component)
                old = assess_visible_skeleton(skeleton)
                new = extract_mask((labels == component).astype(np.uint8), origin['scores'][i], path.stem + '_c' + str(component))
                comparisons.append({'mask': path.name, 'component': component, 'area': area,
                                    'old_eligible': old['geometry_pair_eligible'],
                                    'new_simple_path': new['geometry']['state'] == 'simple_visible_path',
                                    'boundary_truncated': new['boundary_truncated'],
                                    'score': origin['scores'][i], 'old': old, 'new': new['geometry']})
        assessment = assess_view(records, [], origin['image_binding']['image_size'], {'confirmed': False})
        diagnostic = {'run': relative, 'source': str(source), 'origin_manifest_sha256': origin['manifest_sha256'],
                      'reuse_existing_masks': True, 'fresh_sam_this_turn': False,
                      'instance_count': len(records), 'whole_instance_simple_paths': sum(r['geometry']['state']=='simple_visible_path' for r in records),
                      'same_components_count': len(comparisons),
                      'same_components_old_eligible': sum(c['old_eligible'] for c in comparisons),
                      'same_components_new_simple_paths': sum(c['new_simple_path'] for c in comparisons),
                      'newly_simple_components': sum(c['new_simple_path'] and not c['old_eligible'] for c in comparisons),
                      'newly_simple_high_score_not_boundary': sum(c['new_simple_path'] and not c['old_eligible'] and c['score']>=.75 and not c['boundary_truncated'] for c in comparisons),
                      'comparison_records': comparisons, 'selected_scope_assessment_without_confirmed_ports': assessment,
                      'actual_topology_verdict': 'insufficient_evidence', 'real_hit_gain': None,
                      'claim_boundary': 'Geometry diagnostic only, no new terminal identities or actual connection ground truth.'}
        save(OUT / (run.name + '_geometry.json'), diagnostic)
        diagnostics.append({k:v for k,v in diagnostic.items() if k not in ('comparison_records','selected_scope_assessment_without_confirmed_ports')})
    after = source_pins()
    save(OUT / 'report.json', {'status':'complete', 'tests':test.testsRun,
         'constructed_decisions':results, 'real_mask_geometry_diagnostics':diagnostics,
         'mainline_unchanged':pins == after, 'mainline_pins_before':pins, 'mainline_pins_after':after,
         'no_real_accuracy_claim':True, 'deployed':False,
         'next_required':'Confirm actual port identities and traceable expected edges for a bounded visible wire; do not move ports to tips.'})
    if pins != after:
        raise RuntimeError('mainline drift during isolated audit')
    print(json.dumps({'status':'complete', 'tests':test.testsRun, 'constructed_decisions':results,
                      'real_geometry':diagnostics, 'mainline_unchanged':True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
