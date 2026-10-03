"""Read-only sampling-grid audit, independent of any learned head scores."""
import collections
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,'E:/PythonProject10')
from inspection_agent.optional_port_crop_review import sha,read_image
from current_port_baseline_audit import ROOT,load,save
from dense_port_probe import GRID,boxes_from_polygon_labels
DATA=ROOT/'artifacts/port_training_multiscale_20261002/dataset'
OUT=ROOT/'artifacts/dense_patch_geometry_audit_20261003'


def main():
    if OUT.exists():raise FileExistsError('Preserve audit')
    path=DATA/'dataset_manifest.json';manifest=load(path);pins={str(path):sha(path),str(Path(__file__)):sha(Path(__file__))}
    totals={};rows=[]
    for record in manifest['records']:
        split=record['split'];imagepath=DATA/record['image'];labelpath=DATA/record['label']
        assert sha(imagepath)==record['image_sha256'] and sha(labelpath)==record['label_sha256']
        pins[str(imagepath)]=record['image_sha256'];pins[str(labelpath)]=record['label_sha256']
        h,w=read_image(imagepath).shape[:2]
        boxes=boxes_from_polygon_labels(labelpath.read_text(encoding='utf-8'),w,h)
        if split not in totals:totals[split]=dict(crops=0,targets=0,at_least_one_dimension_below_one_token=0,shapes=collections.Counter())
        totals[split]['crops']+=1;totals[split]['targets']+=len(boxes);totals[split]['shapes'][f'{w}x{h}']+=1
        lengths=[min((b['box'][2]-b['box'][0])*GRID/w,(b['box'][3]-b['box'][1])*GRID/h) for b in boxes]
        tiny=sum(value<1 for value in lengths);totals[split]['at_least_one_dimension_below_one_token']+=tiny
        rows.append(dict(image=record['image'],source_image=record['source_image'],split=split,shape=[h,w],
                         targets=len(boxes),minimum_box_dimension_in_tokens=lengths))
    OUT.mkdir();save(OUT/'report.json',dict(status='complete',grid=GRID,summary=totals,cases=rows,pins=pins,
        trained_predictions_not_used=True,labels_unmodified=True,source_or_field_accuracy=False))
    print(str(totals),flush=True)


if __name__=='__main__':main()
