"""Geometry-only action replay plus reused ROI auditor and cached head scores."""
import copy
import statistics
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from audit_fine_consensus_rank import duplicate
import audit_sparse_resolution_geometry as core
from audit_sparse_nomination import independent_actions

SOURCE = ROOT / 'artifacts/sparse_nomination_resolution_20261004'
OUT = ROOT / 'artifacts/sparse_nomination_resolution_replay_20261004'


def geometric_actions(case, probabilities):
    assert probabilities == case['probabilities']
    chosen = independent_actions(case['seeds'], probabilities, case['remaining_budget'])
    assert chosen == case['chosen_seeds']
    return chosen


def verify_case(case, saved):
    prior = core.prescreen
    try:
        core.prescreen = geometric_actions
        result = core.verify_case(case, saved)
    finally:
        core.prescreen = prior
    assert saved['action_selection_uses_no_semantic_scores'] is False
    if saved['proposals']:
        import torch
        torch.set_num_threads(2)
        path = SOURCE / 'train' / (Path(saved['image']).stem+'_features.pt')
        feature = torch.load(path, map_location='cpu', weights_only=True)
        assert feature['source_sha256'] == saved['source_sha256']
        assert feature['boxes'] == [p['box_xyxy'] for p in saved['proposals']]
        x = feature['features']
        assert x.shape == (len(saved['proposals']), 6144) and torch.isfinite(x).all()
        from inspection_agent.paired_native_pose import HEAD_RELATIVE, HEAD_SHA
        assert sha(REPO / HEAD_RELATIVE) == HEAD_SHA == saved['head_sha256']
        state = torch.load(REPO / HEAD_RELATIVE, map_location='cpu', weights_only=True)['state_dict']
        scores = torch.nn.functional.linear(x, state['weight'], state['bias']).softmax(1)
        torch.testing.assert_close(scores, torch.tensor(saved['probabilities']), atol=1e-7, rtol=1e-6)
    return result


def main():
    source = load(SOURCE / 'report.json')
    pins = source['pins']
    prior_source, prior_out, prior_prescreen = core.SOURCE, core.OUT, core.prescreen
    try:
        # Preserve the original auditor. Its metadata/ROI/vote/rank/GT checks
        # are reused, while action selection is independently reconstructed.
        core.SOURCE, core.OUT, core.prescreen = SOURCE, OUT, geometric_actions
        for record in load(core.PREP / 'report.json')['cases']:
            case = load(Path(record['path']))
            saved = load(SOURCE / 'train' / (Path(record['image']).stem+'_predictions.json'))
            if saved['proposals']:
                path = SOURCE / 'train' / (Path(saved['image']).stem+'_features.pt')
                assert sha(path) == pins[str(path)]
            verify_case(case, saved)
        core.main()
        result = load(OUT / 'report.json')
        result.update(semantic_probabilities_recomputed=True,
                      original_pixel_DINO_features_not_independently_reextracted=True,
                      reused_ROI_auditor_sha256=sha(Path(core.__file__)),
                      auditor_sha256=sha(Path(__file__)))
        save(OUT / 'report.json', result)
        print({k: v for k, v in result.items() if k != 'pins'}, flush=True)
    finally:
        core.SOURCE, core.OUT, core.prescreen = prior_source, prior_out, prior_prescreen


if __name__ == '__main__':
    main()
