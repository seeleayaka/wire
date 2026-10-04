"""Frozen paired-head response to identical known-normal reference features.

Diagnostic only: never changes selection or uses GT to choose thresholds.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from inspection_agent.paired_native_pose import HEAD_RELATIVE, HEAD_SHA, native_pose_runtime_fingerprint

SOURCE = ROOT / 'artifacts/evidence_first_resolution_20261004'
AUDIT = ROOT / 'artifacts/evidence_first_resolution_replay_20261004/report.json'
OUT = ROOT / 'artifacts/reference_identity_diagnostic_20261004'


def main():
    import torch
    torch.set_num_threads(2)
    if OUT.exists():
        raise FileExistsError('Preserve frozen reference identity audit')
    report = load(SOURCE / 'report.json')
    audit = load(AUDIT)
    assert audit['status'] == 'pass' and audit['source_report_sha256'] == sha(SOURCE / 'report.json')
    assert all(sha(Path(p)) == v for p, v in report['pins'].items())
    assert native_pose_runtime_fingerprint(REPO) == report['runtime']
    headpath = REPO / HEAD_RELATIVE
    assert sha(headpath) == HEAD_SHA
    state = torch.load(headpath, map_location='cpu', weights_only=True)['state_dict']
    pins = {str(p): sha(p) for p in (Path(__file__), SOURCE / 'report.json', AUDIT, headpath)}
    cases = []
    for path in sorted((SOURCE / 'train').glob('*_predictions.json')):
        pins[str(path)] = sha(path)
        case = load(path)
        n = len(case['proposals'])
        if not n:
            cases.append(dict(image=case['image'], proposals=0, reference_identity_class_counts=[0, 0, 0],
                              reference_identity_confident_positive=0, added_cues=0,
                              added_cues_reference_confident_other=0))
            continue
        featurepath = path.with_name(path.name.replace('_predictions.json', '_features.pt'))
        assert sha(featurepath) == report['pins'][str(featurepath)]
        pins[str(featurepath)] = sha(featurepath)
        data = torch.load(featurepath, map_location='cpu', weights_only=True)
        x = data['features']
        assert x.shape == (n, 6144) and torch.isfinite(x).all()
        assert data['boxes'] == [p['box_xyxy'] for p in case['proposals']]
        assert data['source_sha256'] == case['source_sha256']
        expected = x[:, 1536:3072]
        identity = torch.cat((expected, expected, torch.zeros_like(expected), expected.square()), dim=1)
        with torch.inference_mode():
            observed_scores = torch.nn.functional.linear(x, state['weight'], state['bias']).softmax(1)
            scores = torch.nn.functional.linear(identity, state['weight'], state['bias']).softmax(1)
        torch.testing.assert_close(observed_scores, torch.tensor(case['probabilities']), atol=1e-7, rtol=1e-6)
        classes = scores.argmax(1)
        high = (classes != 0) & (scores.max(1).values >= .98)
        added = case['trial']['all_predictions'][len(case['current']['all_predictions']):]
        identity_rows = []
        for row in added:
            indices = [i for i, p in enumerate(case['proposals'])
                       if p['class_id'] == row['class_id'] and p['box_xyxy'] == row['box_xyxy']]
            assert len(indices) == 1
            i = indices[0]
            identity_rows.append(dict(box=row['box_xyxy'], class_id=row['class_id'],
                                      reference_identity_probabilities=scores[i].tolist()))
        cases.append(dict(image=case['image'], proposals=n,
                          reference_identity_class_counts=torch.bincount(classes, minlength=3).tolist(),
                          reference_identity_confident_positive=int(high.sum()), added_cues=len(added),
                          added_cues_reference_confident_other=sum(r['reference_identity_probabilities'][0] >= .98 for r in identity_rows),
                          additions=identity_rows))
    assert len(cases) == 192 and sum(r['proposals'] for r in cases) == 2002
    assert all(sha(Path(p)) == v for p, v in pins.items())
    OUT.mkdir()
    result = dict(status='complete', sources=192, proposals=2002,
                  reference_identity_class_counts=[sum(r['reference_identity_class_counts'][i] for r in cases) for i in range(3)],
                  reference_identity_confident_positive=sum(r['reference_identity_confident_positive'] for r in cases),
                  added_cues=sum(r['added_cues'] for r in cases),
                  added_cues_reference_confident_other=sum(r['added_cues_reference_confident_other'] for r in cases),
                  cases=cases, pins=pins, no_GT_read=True, no_policy_or_threshold_change=True,
                  cached_feature_identity_not_new_pixel_inference=True, field_accuracy=False)
    save(OUT / 'report.json', result)
    print({k: v for k, v in result.items() if k not in ('cases', 'pins')})


if __name__ == '__main__':
    main()
