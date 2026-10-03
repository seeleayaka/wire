"""Freeze ALL270 all-class graph selections without reading any target boxes."""
import copy
import os
import shutil
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from current_port_baseline_audit import ROOT,REPO,BASE,DATA,load,save,read_current_case
from evaluate_strong_student_feature_consensus import cached_cases
from port_support_graph_policy import append_support_graph,potential_peer_support,POLICY
from inspection_agent.optional_port_crop_review import sha,read_image,predict
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from inspection_agent.feature_residual_port_support import WEIGHT_SHA,WEIGHT_RELATIVE
from inspection_agent.teacher_student_port_support import STUDENT_SHA,STUDENT_RELATIVE,native_selection
from inspection_agent.context_port_recheck import predict_seed_views,recheck_proposals
OUT=ROOT/'artifacts/paired_support_graph_20261003'


def main():
    if (OUT/'source_selections').exists():raise FileExistsError('Preserve selections')
    frozen=resolution_runtime_fingerprint(REPO)
    assert sha(REPO/WEIGHT_RELATIVE)==WEIGHT_SHA and sha(REPO/STUDENT_RELATIVE)==STUDENT_SHA
    paths=[Path(__file__),Path(__file__).with_name('port_support_graph_policy.py'),Path(__file__).with_name('evaluate_strong_student_feature_consensus.py'),
           Path(__file__).with_name('current_port_baseline_audit.py'),OUT.parent/'paired_support_graph_preregistration_20261003/PLAN.md']
    pins={str(p):sha(p) for p in paths};started=time.monotonic();model=None;fresh=0;reuse=0;counts={}
    (OUT/'config/Ultralytics').mkdir(parents=True,exist_ok=True)
    font=OUT/'config/Ultralytics/Arial.ttf'
    if not font.exists():shutil.copy2('C:/Windows/Fonts/arial.ttf',font)
    os.environ.update(YOLO_CONFIG_DIR=str(OUT/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',HF_HUB_OFFLINE='1')
    selections=OUT/'source_selections';selections.mkdir()
    try:
        for stage,total in (('train',192),('inner',48),('outer',30)):
            reportpath=BASE/stage/'report.json';pins[str(reportpath)]=sha(reportpath);entries=load(reportpath)['cases'];assert len(entries)==total
            folder=selections/stage;folder.mkdir();records=[]
            for index,entry in enumerate(entries):
                name=entry['image'];teacher,record=read_current_case(stage,entry,pins);student,peer,real=cached_cases(stage,entry,teacher,pins)
                current=record['trial'];room=len(current['all_predictions'])-len(current['primary'])<5
                assert student['weight_sha256']==STUDENT_SHA and peer['weight_sha256']==WEIGHT_SHA
                assert student['source_sha256']==peer['source_sha256']==teacher['source_sha256']
                if not real and room and potential_peer_support(student):
                    cache=ROOT/'artifacts/port_support_graph_plugs_20261003'/('all_train' if stage=='train' else stage)/(Path(name).stem+'_fresh_peer.json')
                    if cache.exists():
                        peer=load(cache);pins[str(cache)]=sha(cache);reuse+=1
                    else:
                        if model is None:
                            import torch
                            from ultralytics import YOLO
                            torch.set_num_threads(4);native=YOLO(str(REPO/WEIGHT_RELATIVE))
                            assert native.task=='segment' and dict(native.names)=={0:'unplugged_plug',1:'unplugged_jack'}
                            class Capped:
                                def predict(self,*a,**kw):
                                    torch.set_num_threads(4);r=native.predict(*a,**kw);torch.set_num_threads(4);return r
                            model=Capped()
                        image=read_image(DATA/'images'/('val01' if stage=='outer' else 'train01')/name);raw=predict(model,image)
                        peer=dict(image=name,source_sha256=teacher['source_sha256'],weight_sha256=WEIGHT_SHA,predictions=raw,zoom_evidence=[])
                        if len(native_selection(peer)['supplementary'])<5:peer['zoom_evidence']=predict_seed_views(model,image,recheck_proposals(raw))
                        fresh+=1
                    real=True
                assert peer['weight_sha256']==WEIGHT_SHA and peer['source_sha256']==student['source_sha256']
                assert peer['predictions']['source_shape']==student['predictions']['source_shape']==teacher['predictions']['source_shape']
                trial=append_support_graph(current,student,peer);assert trial['strong_consensus_fallback_reason'] is None
                path=folder/(Path(name).stem+'.json')
                save(path,dict(entry=entry,teacher=teacher,current=current,student=student,feature=peer,peer_real=real,trial=trial))
                records.append(dict(image=name,path=str(path),sha256=sha(path),additions=len(trial['strong_consensus_additions'])))
                save(OUT/'preparation_progress.json',dict(status='running',pid=os.getpid(),stage=stage,completed=index+1,total=total,fresh_peer_sources=fresh,reused_verified_peer_sources=reuse,seconds=round(time.monotonic()-started,2)))
            counts[stage]=dict(images=total,with_additions=sum(r['additions']>0 for r in records),additions=sum(r['additions'] for r in records))
            save(folder/'index.json',dict(records=records,summary=counts[stage]))
        assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
        save(selections/'protocol.json',dict(status='complete',pins=pins,runtime_fingerprint=frozen,policy=POLICY,
            selection_before_label_geometry=True,fresh_peer_sources=fresh,reused_verified_peer_sources=reuse,summary=counts,
            no_source_only_failure_relabeled=True,no_automatic_deployment=True,field_accuracy=False))
        save(OUT/'preparation_progress.json',dict(status='complete',summary=counts,seconds=round(time.monotonic()-started,2)))
        print(str(counts),flush=True)
    except BaseException as error:
        save(OUT/'preparation_progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise


if __name__=='__main__':main()
