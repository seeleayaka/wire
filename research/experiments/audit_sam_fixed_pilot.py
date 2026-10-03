"""Fixed one-image SAM geometric ceiling; not a label-selected detector."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize

ROOT = Path('E:/PythonProject10')
sys.path.insert(0, str(ROOT))
from tools.merge_audit import setup
setup(ROOT)  # Torch before Qt on Windows.
from inspection_agent.visible_segment_geometry import assess_visible_skeleton
from evaluate_mendeley_balanced import load_target_boxes, read_image
from tools.evaluate_mendeley_local_refinement import iou
import assembly_auto_review_robust_v3 as perspective


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def component_boxes(mask, mask_id, *, offset=(0, 0)):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    result = []
    for component_id, (x, y, w, h, pixels) in enumerate(stats[1:], 1):
        if pixels < 50: continue
        skeleton = skeletonize(labels[y:y+h, x:x+w] == component_id)
        geometry = assess_visible_skeleton(skeleton)
        result.append({'mask_id': mask_id, 'component_id': component_id,
                       'left': int(x)+offset[0], 'top': int(y)+offset[1],
                       'right': int(x+w)+offset[0], 'bottom': int(y+h)+offset[1],
                       'pixels': int(pixels), 'geometry_pair_eligible': geometry['geometry_pair_eligible'],
                       'candidate_tip_count': geometry['candidate_tip_count'],
                       'branch_cluster_count': geometry['branch_cluster_count']})
    return result


def metrics(boxes, targets):
    best = [max((iou(b, t) for b in boxes), default=0.) for t in targets]
    return {'region_count': len(boxes), 'fragment_count': len(targets),
            'any_bbox_overlap': sum(v > 0 for v in best),
            'iou_ge_01': sum(v >= .1 for v in best), 'iou_ge_05': sum(v >= .5 for v in best),
            'mean_best_iou': float(np.mean(best)) if best else None, 'best_iou_per_fragment': best}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('use fresh output')
    sam = json.loads((args.source_dir/'report.json').read_text(encoding='utf-8'))
    source = Path(sam['input'])
    if source.name != 'disconnected_002.JPG' or source.parent.name != 'val01': raise ValueError('fixed pilot only')
    assert sam['prompt'] == 'cable' and sam['confidence_threshold'] == .4 and not sam['image_state_cache_reused']
    data = ROOT/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    image = read_image(source); reference = read_image(data/'images/train01/normal_073.JPG')
    captured = []; original = cv2.findHomography
    def capture(*a, **kw):
        value = original(*a, **kw)
        captured.append(value[0].copy() if value[0] is not None else None)
        return value
    cv2.findHomography = capture
    try:
        cv2.setRNGSeed(0); aligned, alignment = perspective.automatic_homography(reference, image)
    finally: cv2.findHomography = original
    if aligned is None or len(captured) != 1 or captured[0] is None: raise ValueError('registration failed')
    transform = captured[0]; valid = perspective.auto.LAST_WARP_VALID_MASK > 0
    # Scoring annotations only: never passed into component generation.
    originals = load_target_boxes(data/'labels/val01/disconnected_002.txt', image.shape[1], image.shape[0])
    targets = []
    for l, t, r, b in originals:
        corners = cv2.perspectiveTransform(np.float32([[[l,t],[r,t],[r,b],[l,b]]]), transform).reshape(-1, 2)
        targets.append([int(np.floor(corners[:,0].min())), int(np.floor(corners[:,1].min())),
                        int(np.ceil(corners[:,0].max())), int(np.ceil(corners[:,1].max()))])
    trace = json.loads((ROOT/'output/mendeley_merge_audit_20260928/disconnected_002.json').read_text(encoding='utf-8'))
    if targets != trace['targets']: raise ValueError('transformed scoring rectangles differ from frozen val cache')
    frozen = json.loads((ROOT/'output/mendeley_cnn_qualified_support_20260929/report.json').read_text(encoding='utf-8'))
    case = next(c for c in frozen['results']['cnn_calibrated_support']['cases'] if c['image'] == source.name)
    parents = case['candidates']
    if trace['source_sha256'] != digest(ROOT/'prototype/tiled_dino_review.py'): raise ValueError('DINO source drift')
    masks = []
    for mask_id in range(1, sam['instance_count']+1):
        raw = np.asarray(Image.open(args.source_dir/f'mask_{mask_id:03d}.png').convert('L')) > 0
        assert raw.shape == image.shape[:2] and int(raw.sum()) == sam['mask_pixel_counts'][mask_id-1]
        masks.append(raw)
    union = np.logical_or.reduce(masks)
    source_boxes = [b for n, mask in enumerate(masks, 1) for b in component_boxes(mask, n)]
    stored = np.asarray(Image.open(args.source_dir/'mask_union.png').convert('L')) > 0
    assert np.array_equal(union, stored) and int(union.sum()) == sam['mask_union_pixel_count']
    warped = [cv2.warpPerspective(m.astype(np.uint8), transform, (aligned.shape[1], aligned.shape[0]),
               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool) & valid for m in masks]
    warped_union = np.logical_or.reduce(warped)
    all_boxes = [b for n, mask in enumerate(warped, 1) for b in component_boxes(mask, n)]
    inside = []
    for pidx, parent in enumerate(parents):
        l,t,r,b = (int(parent[k]) for k in ('left','top','right','bottom'))
        l,t,r,b = max(0,l),max(0,t),min(aligned.shape[1],r),min(aligned.shape[0],b)
        if r<=l or b<=t: raise ValueError('bad parent bounds')
        for midx, mask in enumerate(warped, 1):
            for box in component_boxes(mask[t:b,l:r], midx, offset=(l,t)):
                inside.append({'parent_index': pidx, **box})
    # All generated boxes retained; scoring oracle must never choose deployed hints.
    groups = {'frozen_parents': parents, 'all_sam_component_bounds': all_boxes,
              'all_unbranched_sam_component_bounds': [b for b in all_boxes if b['geometry_pair_eligible']],
              'all_sam_parent_intersections': inside,
              'all_unbranched_sam_parent_intersections': [b for b in inside if b['geometry_pair_eligible']],
              'parents_plus_all_sam_intersections': parents+inside}
    scores = {name: metrics(boxes, targets) for name, boxes in groups.items()}
    raw_fractions = [float(union[max(0,t):min(union.shape[0],b),max(0,l):min(union.shape[1],r)].mean())
                     for l,t,r,b in originals]
    args.output.mkdir(parents=True)
    fingerprints = {str(p): digest(p) for p in (source, Path(sam['checkpoint']),
                     ROOT/'manual_review/run_sam3_cable_probe.py', ROOT/'prototype/assembly_auto_review_robust_v3.py',
                     args.source_dir/'report.json', args.source_dir/'mask_union.png', Path(__file__))}
    result = {'image': source.name, 'split': 'val01', 'pilot_count': 1, 'sam': sam,
        'fingerprints': fingerprints, 'alignment': alignment, 'actual_homography': transform.tolist(),
        'source_fragments': originals, 'aligned_fragments': targets, 'cached_fragment_coordinates_exact': True,
        'raw_source_fragment_mask_fraction': raw_fractions,
        'raw_source_component_ceiling': metrics(source_boxes, originals),
        'raw_source_fragments_with_any_mask_pixels': sum(v>0 for v in raw_fractions),
        'source_fragment_count': len(originals), 'metrics': scores, 'generated_geometry': groups,
        'warning': 'All SAM boxes and parent intersections are geometric ceiling diagnostics, not ranked predictions. One fault image, no normal controls, no validation or field accuracy improvement; fragmented labels are not wire masks. Clipped skeleton ends are not physical/electrical cable ends.',
        'formal_path_changed': False}
    (args.output/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    board = Image.new('RGB',(2100,600),'#20252a')
    for column, name in enumerate(('frozen_parents','all_sam_component_bounds','all_sam_parent_intersections')):
        pixels = cv2.cvtColor(aligned,cv2.COLOR_BGR2RGB)
        pixels[warped_union] = (pixels[warped_union].astype(float)*.55+np.array([0,220,210])*.45).astype(np.uint8)
        panel = Image.fromarray(pixels); draw = ImageDraw.Draw(panel)
        for target in targets: draw.rectangle(target, outline='red', width=3)
        for box in groups[name]: draw.rectangle([box[k] for k in ('left','top','right','bottom')], outline='yellow',width=4)
        panel.thumbnail((690,535)); board.paste(panel,(700*column,55))
        draw = ImageDraw.Draw(board); draw.text((700*column+8,8),name,fill='white')
        draw.text((700*column+8,27),'regions='+str(len(groups[name]))+' | geometric diagnostics only',fill='white')
    board.save(args.output/'aligned_comparison.png')
    print(json.dumps({'scores': scores, 'raw_mask_fragment_intersections':sum(v>0 for v in raw_fractions),
                      'target_count':len(targets), 'new_sam_instances':len(masks)},indent=2))


if __name__ == '__main__': main()
