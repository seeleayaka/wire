"""Fixed bounded training recipe; smoke checkpoints never initialize the full run."""
from pathlib import Path

BASE_SHA256='0bac0770b55e5eb5b76a61bc535673288dcec36c2bc0cd25ee0d584c632f3413'


def training_options(data,output,mode):
    if mode not in ('smoke','full'):raise ValueError('mode must be smoke or full')
    return dict(data=str(Path(data)),task='segment',epochs=1 if mode=='smoke' else 6,
        imgsz=960,batch=1,nbs=4 if mode=='smoke' else 64,workers=0,device='cpu',pretrained=True,freeze=10,
        optimizer='SGD',lr0=.002,lrf=.01,cos_lr=False,warmup_epochs=1.,
        amp=False,cache=False,deterministic=True,seed=20260929,patience=6,
        mosaic=0.,close_mosaic=0,degrees=0.,translate=.02,scale=.05,
        fliplr=0.,flipud=0.,plots=False,save=True,val=True,
        project=str(Path(output)/'runs'),name='rectports',exist_ok=False,verbose=True)


def select_smoke_records(records):
    result={}
    for split,pos_count,neg_count in (('train',2,2),('val',1,1)):
        chosen=[];sources=set()
        for positive,count in ((True,pos_count),(False,neg_count)):
            choices=sorted((r for r in records if r['split']==split and (r['label_count']>0)==positive),
                           key=lambda r:(r['source_image'],r['tile_id']))
            taken=0
            for r in choices:
                if r['source_image'] in sources:continue
                sources.add(r['source_image']);chosen.append(r);taken+=1
                if taken==count:break
            if taken!=count:raise ValueError('insufficient distinct-source smoke examples')
        result[split]=chosen
    if {r['source_image'] for r in result['train']}&{r['source_image'] for r in result['val']}:
        raise ValueError('source leakage in smoke subset')
    return result
