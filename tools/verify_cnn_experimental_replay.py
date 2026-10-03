"""Implementation regression against immutable cached results; no tuning or new inference."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.merge_audit import setup
from inspection_agent.experimental_cnn import select_with_optional_cnn


def main():
    _, tiled = setup(ROOT)
    tiled.LARGE_ROI_MAX_CANDIDATES = 6
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    evidence = ROOT/'output/mendeley_cnn_heat_evidence_20260929'
    frozen = load(ROOT/'output/mendeley_cnn_qualified_support_20260929/report.json')
    threshold = frozen['threshold']
    key = lambda rows: sorted(tuple(row[k] for k in ('left','top','right','bottom')) for row in rows)
    scenarios = [('val01', frozen,
                   [ROOT/'output/mendeley_merge_audit_20260928', ROOT/'output/mendeley_normal_evidence_20260928'], evidence/'metric'),
                 ('test01', load(ROOT/'output/mendeley_cnn_frozen_test01_20260929/report.json'),
                   [ROOT/'output/mendeley_cnn_frozen_test01_20260929/traces'], ROOT/'output/mendeley_cnn_frozen_test01_20260929')]
    verified = []
    for split, report, directories, maps in scenarios:
        expected = {c['image']: c['candidates'] for c in report['results']['cnn_calibrated_support']['cases']}
        records = [load(p) for d in directories for p in sorted(d.glob('*.json')) if p.name != 'comparison.json']
        assert len(records) == len(expected) == 30
        for record in records:
            assert record['split'] == split
            if split == 'val01':
                assert record['source_sha256'] == hashlib.sha256((ROOT/'prototype/tiled_dino_review.py').read_bytes()).hexdigest()
            else:
                image = ROOT/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images/test01'/record['image']
                assert record['source_sha256'] == hashlib.sha256(image.read_bytes()).hexdigest()
            t = record['trace']; baseline = t['local_candidates']
            pool = tiled._merge_candidates(copy.deepcopy(t['raw']))
            tiled._annotate_roi_edges(pool, t['width'], t['height'])
            pool, suppressed = tiled._publish_candidates(pool, [])
            assert not suppressed
            with np.load(maps/(Path(record['image']).stem+'_maps.npz'), allow_pickle=False) as data:
                score = data['fusion']
            provider = lambda: (score, threshold, {'mode': 'cached regression only'})
            selected, meta = select_with_optional_cnn(baseline, pool, t['width'], t['height'], provider, enabled=True)
            assert meta['status'] == 'applied', meta
            repeat, _ = select_with_optional_cnn(baseline, pool, t['width'], t['height'], provider, enabled=True)
            assert repeat == selected
            x,y,_,_ = record['bounds']
            absolute = [{**b,'left':b['left']+x,'right':b['right']+x,'top':b['top']+y,'bottom':b['bottom']+y} for b in selected]
            assert absolute == expected[record['image']], record['image']
            def unavailable(): raise RuntimeError('injected unavailable evidence')
            off, meta = select_with_optional_cnn(baseline, pool, t['width'], t['height'], unavailable)
            assert off == baseline and meta['status'] == 'disabled'
            fallback, meta = select_with_optional_cnn(baseline, pool, t['width'], t['height'], unavailable, enabled=True)
            assert fallback == baseline and meta['status'] == 'fallback'
            verified.append({'split':split,'image':record['image'],'exact':True})
    live = load(ROOT/'output/cnn_cli_smoke_20260929/disconnected_011.json')
    expected = next(c['candidates'] for c in frozen['results']['cnn_calibrated_support']['cases'] if c['image']=='disconnected_011.JPG')
    assert live['experimental_cnn']['status'] == 'applied' and key(live['candidates']) == key(expected)
    normal = load(ROOT/'output/cnn_cli_smoke_20260929/normal_001.json')
    assert normal['experimental_cnn']['status'] == 'skipped_upstream' and not normal['candidates']
    output = ROOT/'output/cnn_cli_smoke_20260929/replay_verification.json'
    with output.open('x', encoding='utf-8') as file:
        json.dump({'verified':verified,'count':len(verified),'repeat_exact':True,'off_exact':True,
                   'injected_failure_exact':True,'live_val_exact':True,'normal_gate_skips':True,
                   'warning':'cached implementation regression, not a new test-set experiment'},file,indent=2)
    print('60 cached cases exact; repeat/off/failure exact; live validation matches frozen geometry; normal gate skips.')


if __name__ == '__main__': main()
