"""Independently verify the necessary-condition failure, never claim all5 done."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from run_prompt_contrast import digest, save, verify
from bundle_runtime_pins import source_pins
from audit_semantic_visible_bundle_native import flood_components
from audit_cable_socket_controls import replay_hits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/sam_prompt_confidence_source_20261008'


def main():
    import psutil
    for pid in [18184,12328]:
        assert not psutil.pid_exists(pid), 'original worker/watcher still exists; refuse finalization'
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    verify(protocol['pins'])
    assert source_pins() == protocol['mainline_pins']
    scope = json.loads(Path(protocol['confirmed_scope_path']).read_text(encoding='utf-8'))
    checks = []
    for case_id in ['reference','source_visible_01']:
        case = next(r for r in protocol['cases'] if r['id']==case_id)
        actual = np.asarray(Image.open(case['original_source']['path']).convert('RGB').crop(case['crop_box_xyxy']))
        assert np.array_equal(actual,np.asarray(Image.open(case['fresh_crop']).convert('RGB')))
        partial = json.loads((OUT / (case_id+'_partial_audit.json')).read_text(encoding='utf-8'))
        assert len(partial['rows']) == 6
        for row in partial['rows']:
            dest = OUT / case_id / row['recipe']
            inventory = json.loads((dest / 'inventory.json').read_text(encoding='utf-8'))
            assert digest(dest/'native.npz') == inventory['native_sha256']
            data = np.load(dest/'native.npz',allow_pickle=False)
            assert len(data['scores']) == len(row['records'])
            total = 0
            for index,(mask,score,old) in enumerate(zip(data['masks'],data['scores'],row['records'])):
                png = np.asarray(Image.open(dest / ('mask_%03d.png' % index))) > 0
                assert np.array_equal(mask,png) and mask.shape==actual.shape[:2]
                hits = replay_hits(flood_components(png),png.shape[1],case['crop_box_xyxy'][:2],scope['anchors'],case['anchors'])
                assert sorted((n,tuple(sorted(h.items()))) for n,h in hits)==sorted(
                    (p['pixels'],tuple(sorted(p['hits'].items()))) for p in old['components'])
                ys,xs = np.where(png)
                height,width = png.shape
                boundary = bool(len(xs) and (xs.min()<=1 or ys.min()<=1 or xs.max()>=width-2 or ys.max()>=height-2))
                assert boundary == old['boundary']
                eligible = int(sum(score>=.75 and not boundary and all(v>0 for v in h.values()) for _,h in hits))
                assert eligible == old['eligible'] and float(score)==old['score']
                total += eligible
            assert total == row['eligible']
            checks.append(dict(case=case_id,recipe=row['recipe'],masks=len(data['scores']),eligible=total,
                               threshold_counts=inventory['threshold_counts']))
    assert all(r['eligible']==0 for r in checks if r['case']=='source_visible_01')
    verify(protocol['pins'])
    assert source_pins() == protocol['mainline_pins']
    verdict = dict(status='complete_early_rejection', planned_controls=5, completed_controls=2,
        completed_ids=['reference','source_visible_01'],
        untested_ids=['source_visible_02','source_exposed_01','source_exposed_02'],
        third_encoder_started_but_stopped=True, completed_decoders=12,
        reason='only historically missing visible source control has zero eligible components in all6 recipes',
        strict_source_gain_possible=False, no_further_source_or_demo_inference=True,
        independent_native_replay='PASS', checks=checks, source_pins_unchanged=True,
        actual_visual_review=['reference:cable_box','reference:wire_harness_box','reference:electrical_wire_box',
                             'source_visible_01:cable_box','source_visible_01:wire_harness_box'],
        no_electrical_claim=True,electrical_connections_confirmed=0,deployed=False,not_field_accuracy=True,
        stopped_processes_verified=True, synthetic_implementation_tests=2, generic_gate_tests=3)
    save(OUT/'report.json',verdict)
    save(OUT/'progress.json',dict(status='stopped_early_no_source_gain',completed=verdict['completed_ids'],
                                untested=verdict['untested_ids'],report_sha256=digest(OUT/'report.json')))
    save(OUT/'pipeline_report.json',dict(status='complete_early_rejection',no_inference_running=True,
                                       all_five_controls_completed=False,report_sha256=digest(OUT/'report.json')))
    print(json.dumps({k:v for k,v in verdict.items() if k!='checks'}))


if __name__ == '__main__': main()
