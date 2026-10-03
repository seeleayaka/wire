"""Pre-training descriptor geometry readiness; no labels or recognition scoring."""
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,DATA,load,save,sha,read_image,REFERENCE_SHA
OUT=ROOT/'artifacts/paired_core_ring_coverage_20261004'
TRAINING=ROOT/'artifacts/fine_pose_training_20261004/training'
ALIGNMENTS=ROOT/'artifacts/paired_port_semantics_20261003/features_train'


def main():
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    if OUT.exists():raise FileExistsError('Preserve descriptor readiness')
    sources=sorted(load(ROOT/'artifacts/port_training_multiscale_20261002/protocol.json')['train_sources'])
    samples=load(TRAINING/'samples.json');raw=load(TRAINING/'raw_samples.json')
    refs=DATA/'images/train01/normal_073.JPG';assert sha(refs)==REFERENCE_SHA
    shape=read_image(refs).shape[:2]
    pins={str(p):sha(p) for p in (Path(__file__),TRAINING/'samples.json',TRAINING/'raw_samples.json',refs,ROOT/'artifacts/paired_core_ring_preregistration_20261004/PLAN.md',ROOT/'experiments/paired_core_ring.py')}
    OUT.mkdir();started=time.monotonic();records=[];totals={'training':0,'raw':0};invalid={'training':0,'raw':0}
    for completed,name in enumerate(sources):
        save(OUT/'progress.json',dict(status='running',pid=os.getpid(),completed=completed,total=192,seconds=round(time.monotonic()-started,2)))
        local={kind:[(i,r) for i,r in enumerate(rows) if r['image']==name] for kind,rows in [('training',samples),('raw',raw)]}
        if not any(local.values()):records.append(dict(image=name,cases=[]));continue
        path=ALIGNMENTS/(Path(name).stem+'_source.json');pins[str(path)]=sha(path);aligned=load(path)
        assert aligned['alignment']['alignment_quality']['reliable']
        source=DATA/'images/train01'/name;pins[str(source)]=sha(source);assert aligned['source_sha256']==pins[str(source)]
        inverse=np.linalg.inv(np.asarray(aligned['alignment']['source_to_reference_homography'],np.float64))
        mask=cv2.warpPerspective(np.full(shape,255,np.uint8),inverse,(shape[1],shape[0]),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        mask=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))>0
        cases=[]
        for kind,rows in local.items():
            for index,row in rows:
                l,t,r,b=map(float,row['box']);cx,cy=(l+r)/2,(t+b)/2;fractions=[];counts=[];interiors=[]
                for scale in (1,3):
                    w,h=(r-l)*scale,(b-t)*scale
                    xx,yy=np.meshgrid(cx-w/2+(np.arange(64,dtype=np.float32)+.5)*w/64-.5,cy-h/2+(np.arange(64,dtype=np.float32)+.5)*h/64-.5)
                    domain=np.ones((64,64),bool) if scale==1 else ~((xx+.5>=l)&(xx+.5<r)&(yy+.5>=t)&(yy+.5<b))
                    valid=cv2.remap(mask.astype(np.uint8),xx,yy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
                    valid&=(xx>=-.5)&(yy>=-.5)&(xx<shape[1]-.5)&(yy<shape[0]-.5)&domain
                    fractions.append(float(valid.sum())/int(domain.sum()));counts.append(int(valid.sum()))
                    interiors.append(int(cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8)).sum()))
                usable=min(r-l,b-t)>=2 and min(fractions)>=.85 and counts[0]>=16 and counts[1]>=64 and min(interiors)>0
                totals[kind]+=1;invalid[kind]+=not usable
                if not usable:cases.append(dict(kind=kind,index=index,box=row['box'],coverage=fractions,counts=counts,gradient_support=interiors))
        records.append(dict(image=name,cases=cases))
    assert totals=={'training':9808,'raw':969} and all(sha(Path(p))==v for p,v in pins.items())
    result=dict(status='complete',total=totals,invalid=invalid,cases=records,pins=pins,seconds=round(time.monotonic()-started,2),
        no_labels_used=True,no_training=True,no_deployment=True,field_accuracy=False)
    save(OUT/'report.json',result);save(OUT/'progress.json',dict(status='complete',seconds=result['seconds']));print({k:v for k,v in result.items() if k not in ('cases','pins')},flush=True)


if __name__=='__main__':main()
