"""Fixed first-per-kind illustrations of a completed frozen evaluation."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    authorized=Path(__file__).resolve().parent
    output=a.output.resolve()
    if not output.is_relative_to(authorized): raise ValueError('output outside authorized experiments directory')
    if output.exists(): raise FileExistsError('use a fresh illustration directory')
    sys.path.insert(0,str(a.repo)); sys.path.insert(0,str(a.repo/'prototype'))
    import torch  # Must precede Qt imports on this Windows installation.
    from evaluate_mendeley_balanced import read_image
    from tools.diagnose_spatial_hotspots import panel,top_mask
    report=json.loads(a.report.read_text(encoding='utf-8'))
    assert report['status']=='complete'
    cases={m:{r['image']:r for r in d['cases']} for m,d in report['results'].items()}
    audits={r['image']:r for r in report['audits']}
    selected={k:min(name for name in audits if name.startswith(k+'_')) for k in ('damaged','disconnected','misrouted','normal')}
    dataset=a.repo/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    reference=read_image(dataset/'images/train01/normal_073.JPG')
    output.mkdir(parents=True)
    for name in selected.values():
        trace=audits[name]; bounds=trace['bounds']; x,y,right,bottom=bounds
        image=cv2.warpPerspective(read_image(dataset/'images/test01'/name),np.array(trace['actual_homography']),
                                   (reference.shape[1],reference.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        frames=[]
        for mode in ('baseline','cnn_calibrated_support'):
            row=cases[mode][name]
            frame=panel(image,None,row['targets'],bounds,mode+'; cyan=source, orange=candidate')
            h,w=frame.shape[0]-35,frame.shape[1]
            for number,b in enumerate(row['candidates'],1):
                l,t=int((b['left']-x)*w/(right-x)),35+int((b['top']-y)*h/(bottom-y))
                r,bt=int((b['right']-x)*w/(right-x)),35+int((b['bottom']-y)*h/(bottom-y))
                cv2.rectangle(frame,(l,t),(r,bt),(0,165,255),2)
                cv2.putText(frame,str(number),(l,max(45,t)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,165,255),1)
            frames.append(frame)
        with np.load(a.report.parent/(Path(name).stem+'_maps.npz'),allow_pickle=False) as data:
            frames.append(panel(image,top_mask(data['fusion']),cases['baseline'][name]['targets'],bounds,'CNN top5% (red); not detector threshold'))
        cv2.imencode('.png',np.concatenate(frames,axis=1))[1].tofile(str(output/(Path(name).stem+'.png')))
    print(json.dumps({'selected':selected,'output':str(output)},indent=2))


if __name__=='__main__': main()
