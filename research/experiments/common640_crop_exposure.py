"""TRAIN-only geometric exposure audit, never crop selection or detection scoring."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, DATA, load, save, sha
from inspection_agent.port_tiling import tile_windows, near_artificial_edge

SOURCE = ROOT/'artifacts/allport480_teacher_source_20261005'
MISSES = ROOT/'artifacts/allport480_missing_class_coverage_20261005/report.json'
OUT = ROOT/'artifacts/allport_common640_crop_exposure_20261005'


def surviving_tiles(box, width, height, size, stride, margin=16):
    left, top, right, bottom = box
    result = []
    for index, window in enumerate(tile_windows(width, height, size, stride)):
        x, y, r, b = window
        if not x <= left < right <= r or not y <= top < bottom <= b: continue
        local = [left-x, top-y, right-x, bottom-y]
        if not near_artificial_edge(local, window, width, height, margin): result.append(index)
    return result


def main():
    if OUT.exists(): raise FileExistsError('preserve fixed crop exposure diagnosis')
    report = load(SOURCE/'report.json'); missing = load(MISSES)
    if not report['qualifies'] or len(report['cases']) != 192: raise ValueError('complete teacher SOURCE required')
    rows = []; pins = {str(SOURCE/'report.json'): sha(SOURCE/'report.json'), str(MISSES): sha(MISSES), str(Path(__file__)): sha(__file__)}
    for item in report['cases']:
        name = item['image']; label = DATA/'labels/train01'/(Path(name).stem+'.txt')
        pins[str(label)] = sha(label)
        if pins[str(label)] != report['pins'][str(label)]: raise ValueError('original TRAIN label drift')
        index = 0
        for line in label.read_text(encoding='utf-8').splitlines():
            cls, x, y, w, h = map(float, line.split())
            if cls not in (3, 4): continue
            box = [(x-w/2)*3648, (y-h/2)*2736, (x+w/2)*3648, (y+h/2)*2736]
            views = {str(size): surviving_tiles(box, 3648, 2736, size, stride)
                     for size, stride in [(1280, 960), (960, 720), (640, 480)]}
            rows.append(dict(image=name, target_index=index, class_id=int(cls)-3, box_xyxy=box, containing_surviving_tiles=views))
            index += 1
    if len(rows) != 344: raise ValueError('full original TRAIN344 target population required')
    if any(sha(p) != d for p, d in pins.items()): raise ValueError('fixed geometric diagnosis input drift')
    OUT.mkdir()
    save(OUT/'report.json', dict(status='complete', source_images=192, original_targets=344, pins=pins, cases=rows,
         summary={str(size): sum(bool(r['containing_surviving_tiles'][str(size)]) for r in rows) for size in (1280, 960, 640)},
         uses_original_TRAIN_GT_for_diagnosis_only=True, no_GT_seeded_runtime_crops=True, no_model_inference=True,
         no_development_reads=True, no_scale_or_stride_change=True, no_deployment=True, field_accuracy=None,
         warning='Geometric crop exposure only. A fully contained target may remain invisible/undetected; this is NOT recognition accuracy.'))
    print(dict(status='complete', original_targets=344, crop_exposure_only=True))


if __name__ == '__main__': main()
