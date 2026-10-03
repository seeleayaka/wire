"""Predeclared no-aspect control: same3134 rows, exact rectangular vectors."""
import copy
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_rectcontext import OUT as ORIGINAL
from prepare_paired_port_semantics import ROOT, load, save, sha
OUT = ROOT / 'artifacts/paired_rectcontext_original_rows_20261003'


def main():
    if OUT.exists(): raise FileExistsError('Preserve rectangle no-aspect control')
    source = ORIGINAL / 'features_train'; prepared = load(source / 'report.json'); metadata = load(source / 'samples.json')
    assert prepared['samples'] == 4166 and prepared['gt_targets'] == 344
    assert sha(source / 'features.pt') == prepared['aggregate_feature_sha256'] and sha(source / 'samples.json') == prepared['samples_sha256']
    import torch
    data = torch.load(source / 'features.pt', map_location='cpu', weights_only=True)
    indices = [i for i, row in enumerate(metadata) if row['kind'] != 'synthetic_aspect']
    samples = [metadata[i] for i in indices]; assert len(samples) == 3134 and sum(r['kind'] == 'gt_port' for r in samples) == 344
    folder = OUT / 'features_train'; folder.mkdir(parents=True)
    vectors = {key: value[indices] for key, value in data.items()}; torch.save(vectors, folder / 'features.pt'); save(folder / 'samples.json', samples)
    report = copy.deepcopy(prepared); report['pins'].update({str(p): sha(p) for p in (Path(__file__), source / 'report.json', source / 'features.pt', source / 'samples.json')})
    report.update(samples=3134, counts=torch.bincount(vectors['labels'], minlength=3).tolist(),
        aggregate_feature_sha256=sha(folder / 'features.pt'), samples_sha256=sha(folder / 'samples.json'),
        original3134_only=True, excluded_only_synthetic_aspect=True, no_new_encoder_inference=True, no_deployment=True)
    save(folder / 'report.json', report)
    print(str(dict(samples=3134, counts=report['counts'])), flush=True)


if __name__ == '__main__': main()
