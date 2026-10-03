"""ALL192 TRAIN photometric invariance audit; no labels, fitting or detector."""
import os
import sys
import time
from pathlib import Path
from collections import Counter
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from inspection_agent.paired_port_features import expected_in_source
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
from robust_port_exposure import compensate
SOURCE=ROOT/'artifacts/paired_port_semantics_20261003/features_train'
OUT=ROOT/'artifacts/port_exposure_invariance_train_20261004'
PLAN=ROOT/'artifacts/robust_port_exposure_prototype_20261004/PLAN.md'

def main():
    if OUT.exists():raise FileExistsError('Preserve photometric invariance evidence')
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    groupspath=ROOT/'artifacts/port_training_multiscale_20261002/protocol.json';names=sorted(load(groupspath)['train_sources']);assert len(names)==192
    frozen=native_pose_runtime_fingerprint(REPO);referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    reference=read_image(referencepath);OUT.mkdir();started=time.monotonic();rows=[];counts=Counter()
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('robust_port_exposure.py'),groupspath,referencepath,PLAN)}
    try:
        for i,name in enumerate(names):
            source=DATA/'images/train01'/name;path=SOURCE/(Path(name).stem+'_source.json');pins[str(source)]=sha(source);pins[str(path)]=sha(path)
            record=load(path);assert record['source_sha256']==pins[str(source)]
            save(OUT/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),completed=i,total=192,image=name))
            alignment=record['alignment']
            if not alignment.get('alignment_quality',{}).get('reliable'):
                counts['unreliable_registration']+=1;rows.append(dict(image=name,status='abstained_registration'));continue
            original=read_image(source);expected,mask=expected_in_source(reference,alignment['source_to_reference_homography'],original.shape[:2])
            dark=np.rint(original.astype(np.float32)*.85).clip(0,255).astype(np.uint8)
            a,first=compensate(original,expected,mask);b,second=compensate(dark,expected,mask)
            # Uniform audit samples, not known-positive regions.
            valid=mask[::32,::32]>.99;before=float(np.abs(dark[::32,::32].astype(float)-original[::32,::32]).mean(axis=2)[valid].mean())
            after=float(np.abs(a[::32,::32].astype(float)-b[::32,::32]).mean(axis=2)[valid].mean())
            status='both_usable' if first['status']!='abstained' and second['status']!='abstained' else 'at_least_one_abstained'
            counts[status]+=1
            rows.append(dict(image=name,status=status,original_policy=first,dark_policy=second,
                original_exposure_difference_MAE=before,normalized_pair_difference_MAE=after,
                relative_difference=after/before if before else None,not_recognition_metrics=True))
        assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen
        usable=[row for row in rows if row['status']=='both_usable']
        result=dict(status='complete',train_sources=192,counts=dict(counts),
            median_relative_difference=float(np.median([r['relative_difference'] for r in usable])) if usable else None,
            cases=rows,pins=pins,runtime=frozen,no_label_files_read=True,no_fitting=True,no_detector_inference=True,
            no_physical_fault_or_recognition_accuracy=True,no_deployment=True,seconds=round(time.monotonic()-started,2))
        save(OUT/'report.json',result);save(OUT/'progress.json',dict(status='complete',seconds=result['seconds'],counts=dict(counts)))
        print({k:result[k] for k in ('status','counts','median_relative_difference','seconds')},flush=True)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise

if __name__=='__main__':main()
