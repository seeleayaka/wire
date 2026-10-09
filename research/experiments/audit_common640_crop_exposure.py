"""Independent TRAIN GT/window exposure decoding. No detection claims."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, DATA, load, save, sha

SOURCE = ROOT/'artifacts/allport_common640_crop_exposure_20261005/report.json'
MISSES = ROOT/'artifacts/allport480_missing_class_coverage_20261005/report.json'
OUT = ROOT/'artifacts/allport_common640_crop_exposure_audit_20261005'


def exposure(box, width, height, size, stride):
    def axis(length):
        end = max(0, length-size)
        values = list(range(0, end+1, stride))
        return values if values[-1] == end else values+[end]
    kept = []; index = 0
    for y in axis(height):
        for x in axis(width):
            r, b = min(width, x+size), min(height, y+size)
            l, t, rr, bb = box
            inside = x <= l < rr <= r and y <= t < bb <= b
            cut = ((x > 0 and l-x <= 16) or (y > 0 and t-y <= 16)
                   or (r < width and rr-x >= r-x-16) or (b < height and bb-y >= b-y-16))
            if inside and not cut: kept.append(index)
            index += 1
    return kept


def main():
    if OUT.exists(): raise FileExistsError('preserve independent exposure replay')
    digest = sha(SOURCE); report = load(SOURCE); misses = load(MISSES)
    if report['status'] != 'complete' or any(sha(p) != d for p, d in report['pins'].items()):
        raise ValueError('complete pinned source exposure audit required')
    # Include normal cases with zero GT targets; target-row names alone cannot
    # represent the complete image population.
    parentpath = ROOT/'artifacts/allport480_teacher_source_20261005/report.json'
    if report['pins'].get(str(parentpath)) != sha(parentpath): raise ValueError('TRAIN source cohort not bound')
    images = sorted(r['image'] for r in load(parentpath)['cases'])
    if len(images) != 192 or len(set(images)) != 192: raise ValueError('complete unique TRAIN192 inventory required')
    expected = []
    for name in images:
        label = DATA/'labels/train01'/(Path(name).stem+'.txt'); index = 0
        if sha(label) != report['pins'][str(label)]: raise ValueError('GT changed')
        for line in label.read_text(encoding='utf-8').splitlines():
            cls, cx, cy, w, h = map(float, line.split())
            if cls not in (3, 4): continue
            box = [(cx-w/2)*3648, (cy-h/2)*2736, (cx+w/2)*3648, (cy+h/2)*2736]
            expected.append(dict(image=name, target_index=index, class_id=int(cls)-3, box_xyxy=box,
                  containing_surviving_tiles={str(s): exposure(box, 3648, 2736, s, step)
                                             for s, step in [(1280, 960), (960, 720), (640, 480)]}))
            index += 1
    # Normal images have no target rows; bind their original192 membership via the
    # source report, not the set of positive image names.
    allnames = images
    if len(allnames) != 192 or set(images)-set(allnames): raise ValueError('TRAIN cohort drift')
    indexed = {(r['image'], r['target_index']): r for r in report['cases']}
    expectedmap = {(r['image'], r['target_index']): r for r in expected}
    if expectedmap != indexed or len(indexed) != 344: raise ValueError('independent GT/window replay mismatch')
    subset = []
    for missing in misses['cases']:
        row = indexed[(missing['image'], missing['target_index'])]
        if row['class_id'] != missing['class_id'] or any(abs(a-b) > 1e-7 for a,b in zip(row['box_xyxy'], missing['GT_box'])):
            raise ValueError('eligible-miss identity differs from original GT')
        subset.append(row)
    if len(subset) != 14: raise ValueError('fixed source eligible14 miss population required')
    if sha(SOURCE) != digest or any(sha(p) != d for p,d in report['pins'].items()): raise ValueError('exposure evidence drift')
    summary = {str(s): sum(bool(r['containing_surviving_tiles'][str(s)]) for r in expected) for s in (1280,960,640)}
    if summary != report['summary']: raise ValueError('exposure count mismatch')
    OUT.mkdir(); save(OUT/'report.json', dict(status='pass', source_report_sha256=digest, auditor_sha256=sha(__file__),
          original_targets=344, summary=summary, eligible14_summary={str(s): sum(bool(r['containing_surviving_tiles'][str(s)]) for r in subset) for s in (1280,960,640)},
          eligible14_cases=subset, no_model_inference=True, no_crop_policy_change=True, no_development_reads=True,
          no_deployment=True, field_accuracy=None, warning='Geometric exposure, not detectability or recognition accuracy.'))
    print(dict(status='pass', summary=summary, eligible14_common640=sum(bool(r['containing_surviving_tiles']['640']) for r in subset)))


if __name__ == '__main__': main()
