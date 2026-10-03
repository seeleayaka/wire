"""Post-completion diagnostics only; never search a threshold or select boxes."""
import collections
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import OUT, load, save, sha
from port_semantic_verifier import PROBABILITY_GATE


def main():
    import torch
    source = OUT / 'features_train'
    prepared = load(source / 'report.json')
    trained = load(OUT / 'heads_oof/report.json')
    assert prepared['status'] == trained['status'] == 'complete'
    assert sha(source / 'samples.json') == prepared['samples_sha256']
    probabilities_path = OUT / 'heads_oof/oof_probabilities.pt'
    assert sha(probabilities_path) == trained['probabilities_sha256']
    samples = load(source / 'samples.json')
    probabilities = torch.load(probabilities_path, map_location='cpu', weights_only=True)
    scores, predictions = probabilities.max(dim=1)
    accepted = (scores >= PROBABILITY_GATE) & (predictions > 0)
    result = {}
    for kind in sorted({r['kind'] for r in samples}):
        indices = [i for i, r in enumerate(samples) if r['kind'] == kind]
        confusion = [[0]*3 for _ in range(3)]
        correct, unmatched, wrong_class, below_gate = 0, 0, 0, 0
        for i in indices:
            truth, predicted = samples[i]['label'], int(predictions[i])
            confusion[truth][predicted] += 1
            if accepted[i]:
                correct += truth == predicted
                unmatched += truth != predicted
            if truth > 0:
                wrong_class += truth != predicted
                below_gate += truth == predicted and not accepted[i]
        result[kind] = dict(samples=len(indices), accepted_correct=correct,
                            accepted_unmatched=unmatched, argmax_confusion=confusion,
                            positive_wrong_argmax=wrong_class, positive_correct_below_fixed_gate=below_gate)
    sources = prepared['sources']
    diagnostics = dict(status='complete', threshold=PROBABILITY_GATE,
        no_threshold_search=True, diagnostic_only=True, classifier_OOF_only=True,
        field_accuracy=False, by_kind=result,
        source_status=dict(collections.Counter(r['status'] for r in sources)),
        empty_feature_sources=[r['image'] for r in sources if r.get('samples', 0) == 0],
        valid_GT=sum(r.get('gt_valid', 0) for r in sources),
        invalid_or_abstained_GT=prepared['gt_targets']-prepared['gt_valid'],
        folds=trained['folds'], pins={str(source/'report.json'): sha(source/'report.json'),
            str(OUT/'heads_oof/report.json'):sha(OUT/'heads_oof/report.json'),
            str(probabilities_path):sha(probabilities_path),str(source/'samples.json'):sha(source/'samples.json')})
    destination = OUT / 'classifier_diagnostics'
    destination.mkdir(exist_ok=False)
    save(destination / 'report.json', diagnostics)
    print(str(diagnostics))


if __name__ == '__main__':
    main()
