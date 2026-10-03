"""Read-only visual audit assets for all 245 original val01 fault labels."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "artifacts/mendeley_val_label_audit_20260930_v2"
DATA = Path(r"E:\PythonProject10\data\external_datasets\mendeley_electrical_wiring_faults\Predictive Maintenance for Electrical Wiring Faults")
NAMES = {
    "damaged": ("damaged_007", "damaged_012", "damaged_020", "damaged_031", "damaged_035"),
    "disconnected": ("disconnected_002", "disconnected_006", "disconnected_011", "disconnected_027", "disconnected_048"),
    "misrouted": ("misrouted_012", "misrouted_025", "misrouted_034", "misrouted_040", "misrouted_050"),
}
COLORS = {1: (10, 180, 255), 2: (0, 255, 100), 3: (255, 100, 30), 4: (245, 30, 245)}
CARD_W, CARD_H = 290, 290


def write_jpeg(path, image):
    ok, blob = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    blob.tofile(str(path))


def read_jpeg(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None, path
    return image


def labels(path, width, height):
    result = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        assert len(parts) == 5, (path, index)
        cls, cx, cy, w, h = map(float, parts)
        assert all(math.isfinite(x) for x in (cls, cx, cy, w, h))
        assert cls == int(cls) and int(cls) in (1, 2, 3, 4)
        assert 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1
        x0 = int(round((cx - w/2)*width)); y0 = int(round((cy - h/2)*height))
        x1 = int(round((cx + w/2)*width)); y1 = int(round((cy + h/2)*height))
        assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, (path, index, x0, y0, x1, y1)
        result.append({"index": index, "class": int(cls), "box": [x0, y0, x1, y1], "raw": line})
    return result


def card(image, label, stem):
    x0,y0,x1,y1 = label["box"]
    cx,cy = (x0+x1)/2,(y0+y1)/2
    span = max(260, int(max(x1-x0,y1-y0)*1.45))
    l=max(0,int(cx-span/2)); t=max(0,int(cy-span/2))
    r=min(image.shape[1],l+span); b=min(image.shape[0],t+span)
    l=max(0,r-span); t=max(0,b-span)
    crop=image[t:b,l:r].copy()
    col=COLORS[label["class"]]
    cv2.rectangle(crop,(x0-l,y0-t),(x1-l,y1-t),col,max(2,span//140))
    crop=cv2.resize(crop,(CARD_W,CARD_H-28),interpolation=cv2.INTER_AREA if span>CARD_W else cv2.INTER_LINEAR)
    canvas=np.full((CARD_H,CARD_W,3),26,np.uint8)
    canvas[28:]=crop
    title=f"{stem} #{label['index']:02d} c{label['class']} ({x1-x0}x{y1-y0})"
    cv2.putText(canvas,title,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
    return canvas


def main():
    if ROOT.exists():
        raise FileExistsError(ROOT)
    ROOT.mkdir(parents=True)
    counts=Counter(); by_kind=defaultdict(Counter); records=[]
    for kind, stems in NAMES.items():
        for stem in stems:
            image=read_jpeg(DATA / "images/val01" / (stem+".JPG"))
            height,width=image.shape[:2]
            assert (width,height)==(3648,2736)
            boxes=labels(DATA / "labels/val01" / (stem+".txt"),width,height)
            overlay=image.copy()
            tiles=[]
            for item in boxes:
                counts[item["class"]]+=1; by_kind[kind][item["class"]]+=1
                l,t,r,b=item["box"]
                cv2.rectangle(overlay,(l,t),(r,b),COLORS[item["class"]],5)
                cv2.putText(overlay,f"{item['index']}:c{item['class']}",(l,max(28,t-8)),
                            cv2.FONT_HERSHEY_SIMPLEX,.85,COLORS[item["class"]],2,cv2.LINE_AA)
                tiles.append(card(image,item,stem))
                records.append({"image":stem+".JPG", **item})
            overview=cv2.resize(overlay,(1824,1368),interpolation=cv2.INTER_AREA)
            write_jpeg(ROOT / (stem+"_overview.jpg"),overview)
            cols=4
            rows=[]
            for start in range(0,len(tiles),cols):
                row=tiles[start:start+cols]
                while len(row)<cols:row.append(np.full_like(tiles[0],26))
                rows.append(np.hstack(row))
            sheet=np.vstack(rows)
            write_jpeg(ROOT / (stem+"_details.jpg"),sheet)
            print(f"{stem}: {len(boxes)} labels, class counts={dict(Counter(x['class'] for x in boxes))}",flush=True)
    assert len(records)==245, len(records)
    report={"total":len(records),"class_counts":dict(counts),"by_kind":{k:dict(v) for k,v in by_kind.items()},
            "geometry_checks":"all finite, five values, in bounds, positive area, source image dimensions 3648x2736",
            "scope":"source val01 only, 15 fault images; overview and each label local context rendered",
            "manual_semantic_correctness":"not determined by this script", "records":records}
    (ROOT / "geometry_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="records"},ensure_ascii=False,indent=2))


if __name__ == "__main__":main()
