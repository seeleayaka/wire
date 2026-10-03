"""Audit both square and contained-rectangle averages; subset is not proof."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, DATA, OUT as ORIGINAL, load, save, sha, read_image
from paired_port_semantics import expected_in_source, context_valid_fraction
from paired_rectcontext_features import crop_signature
OUT = ROOT / 'artifacts/paired_rectcontext_coverage_audit_20261003'


def fraction(signature, mask):
    l, t, r, b = signature; h, w = mask.shape
    part = mask[max(0, t):min(h, b), max(0, l):min(w, r)] if r > 0 and b > 0 and l < w and t < h else mask[:0, :0]
    return float(part.sum()) / max(1, (r - l) * (b - t))


def main():
    if OUT.exists(): raise FileExistsError('Preserve independent coverage audit')
    import cv2
    cv2.setNumThreads(2)
    sample_path = ROOT / 'artifacts/paired_shape_train_probe_20261003/samples.json'
    rows = load(sample_path); assert len(rows) == 4166
    pins = {str(p): sha(p) for p in (Path(__file__), Path(__file__).with_name('paired_rectcontext_features.py'), sample_path)}
    reference_path = DATA / 'images/train01/normal_073.JPG'; pins[str(reference_path)] = sha(reference_path); reference = read_image(reference_path)
    records = []; invalid = []; OUT.mkdir()
    for name in sorted({r['image'] for r in rows}):
        local = [r for r in rows if r['image'] == name]
        record_path = ORIGINAL / 'features_train' / (Path(name).stem + '_source.json'); pins[str(record_path)] = sha(record_path); source = load(record_path)
        image_path = DATA / 'images/train01' / name; assert sha(image_path) == source['source_sha256']; pins[str(image_path)] = sha(image_path)
        image = read_image(image_path); _, mask = expected_in_source(reference, source['alignment']['source_to_reference_homography'], image.shape[:2])
        for row in local:
            square = [context_valid_fraction(row['box'], mask, s) for s in (1.5, 3.)]
            rect = [fraction(crop_signature(row['box'], s), mask) for s in (1.5, 3.)]
            assert min(square) >= .85
            result = dict(image=name, kind=row['kind'], label=row['label'], square=square, rectangular=rect)
            records.append(result)
            if min(rect) < .85: invalid.append(result)
        save(OUT / 'progress.json', dict(status='running', completed_samples=len(records), total=4166))
    assert len(records) == 4166 and {p: sha(Path(p)) for p in pins} == pins
    report = dict(status='complete', samples=4166, rectangular_below85=len(invalid), invalid=invalid,
        rectangular_minimum=min(min(r['rectangular']) for r in records), square_minimum=min(min(r['square']) for r in records),
        pins=pins, contained_rectangle_average_not_mathematically_guaranteed=True,
        no_model_inference=True, TRAIN_only=True, no_deployment=True, field_accuracy=False)
    save(OUT / 'report.json', report); save(OUT / 'progress.json', dict(status='complete', rectangular_below85=len(invalid)))
    print(str({k: v for k, v in report.items() if k not in ('pins', 'invalid')}), flush=True)


if __name__ == '__main__': main()
