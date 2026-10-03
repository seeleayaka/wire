"""Readonly smoke paired crops: diagnostic artifacts, not metric acceptance."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, OUT, DATA, load, sha, read_image
from paired_port_semantics import expected_in_source
from port_semantic_verifier import context_box


def main():
    from PIL import Image, ImageDraw
    smoke = OUT / 'smoke'
    protocol = load(smoke / 'protocol.json')
    assert {p: sha(Path(p)) for p in protocol['pins']} == protocol['pins']
    rows = load(smoke / 'samples.json')
    reference = read_image(DATA / 'images/train01/normal_073.JPG')
    tiles = []
    for name in protocol['train_sources']:
        source = read_image(DATA / 'images/train01' / name)
        record = load(smoke / (Path(name).stem + '_source.json'))
        expected, mask = expected_in_source(reference, record['alignment']['source_to_reference_homography'], source.shape[:2])
        chosen = []
        for kind in ('gt_port', 'novel_weak_proposal'):
            chosen.extend([r for r in rows if r['image'] == name and r['kind'] == kind][:2])
        chosen.extend([r for r in rows if r['image'] == name and r['kind'] not in ('gt_port', 'reference_self', 'novel_weak_proposal')][:1])
        for row in chosen:
            tile = Image.new('RGB', (640, 360), 'white')
            draw = ImageDraw.Draw(tile)
            draw.text((8, 8), name + ' / ' + row['kind'] + ' / label=' + str(row['label']), fill='black')
            for index, array in enumerate((source, expected)):
                l, t, r, b = context_box(row['box'], 3.0)
                region = Image.fromarray(array[:, :, ::-1]).crop((int(l), int(t), int(r), int(b)))
                size = region.size
                region.thumbnail((304, 300))
                left = 8 + index * 320 + (304 - region.width) // 2
                top = 45 + (300 - region.height) // 2
                tile.paste(region, (left, top))
                scale_x, scale_y = region.width / size[0], region.height / size[1]
                x1, y1, x2, y2 = row['box']
                draw.rectangle((left + (x1-int(l))*scale_x, top + (y1-int(t))*scale_y,
                                left + (x2-int(l))*scale_x, top + (y2-int(t))*scale_y), outline='red', width=2)
                draw.text((8 + index * 320, 28), 'OBSERVED' if index == 0 else 'ALIGNED EXPECTED', fill='black')
            tiles.append(tile)
    assert tiles
    destination = OUT / 'visual_audit'
    destination.mkdir(exist_ok=False)
    sheet = Image.new('RGB', (1280, 360 * ((len(tiles)+1)//2)), '#dddddd')
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 2)*640, (i//2)*360))
    path = destination / 'smoke_observed_expected.jpg'
    sheet.save(path, quality=92)
    print(str(path))


if __name__ == '__main__':
    main()
