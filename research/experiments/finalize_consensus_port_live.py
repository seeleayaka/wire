"""Recover only final scoring from six already completed inference artifacts."""
import copy,json,sys,unittest
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/consensus_port_live_20261002'
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
import torch
from core_port_resolution_ab_20261002 import DATA,score
from inspection_agent.optional_port_crop_review import sha
from inspection_agent.port_rescue_gui import POLICY_ID
import cv2,numpy as np

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

def main():
    if (OUT/'report.json').exists():raise FileExistsError('Do not overwrite final report')
    protocol=load(OUT/'protocol.json');rows=[];protected={};before_sha={}
    prior=load(ROOT/'artifacts/rescue_mainline_ab_20261002_v2/report.json')
    sources={c['image']:Path(c['output'])/'report_off.json' for c in prior['cases']}
    for name in protocol['gui_sources']:
        stem=Path(name).stem;report_file=OUT/stem/'report.json';report=load(report_file)
        summary=report['independent_port_rescue'];evidence=Path(summary['evidence_path'])/'evidence.json'
        saved=load(evidence);result=saved['result']
        assert saved['binding']['policy']==POLICY_ID and saved['binding']['supplementary'] is True
        assert report['review_regions']==load(sources[name])['review_regions']
        previous=load(ROOT/'artifacts/precision_port_gui_20261002_v2'/stem/'report.json')['independent_port_rescue']
        assert result['rescue_hints']==previous['rescue_hints']
        protected[str(evidence)]=sha(evidence);protected[str(report_file)]=sha(report_file)
        rows.append(dict(image=name,mode='actual_gui',report=report,result=result,evidence=str(evidence)))
    for name in protocol['new_initial_sources']:
        stem=Path(name).stem;files=list((OUT/(stem+'_new_initial')).glob('*/consensus_evidence.json'));assert len(files)==1
        evidence=files[0];report_file=evidence.parent/'initial_report.json'
        report=load(report_file);result=load(evidence)
        assert report['decision']=='sam3_fusion_running' and result['parents']==report['review_regions']
        protected[str(evidence)]=sha(evidence);protected[str(report_file)]=sha(report_file)
        rows.append(dict(image=name,mode='fresh_initial_without_sam',report=report,result=result,evidence=str(evidence),sam_fusion_pending=True))
    # No predictions or selection recomputed; only score the preserved boxes.
    for row in rows:
        result=row.pop('result');report=row.pop('report');matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64)
        split='test01' if row['mode']=='actual_gui' else 'train01';targets=[]
        assert sha(report['inspection'])==report['image_fingerprints']['source_sha256']
        assert sha(report['reference'])==report['image_fingerprints']['reference_sha256']
        assert len(result['rescue_hints'])<=5 and len(result['supplementary_hints'])<=5 and not result['automatic_fault_verdict']
        for line in (DATA/'labels'/split/(Path(row['image']).stem+'.txt')).read_text(encoding='utf-8').splitlines():
            cls,cx,cy,w,h=map(float,line.split())
            if cls not in (3,4):continue
            l,t,r,b=(cx-w/2)*3648,(cy-h/2)*2736,(cx+w/2)*3648,(cy+h/2)*2736
            q=cv2.perspectiveTransform(np.float32([[[l,t],[r,t],[r,b],[l,b]]]),matrix).reshape(-1,2)
            targets.append(dict(class_id=int(cls)-3,box=[q[:,0].min(),q[:,1].min(),q[:,0].max(),q[:,1].max()]))
        adapt=lambda hints:[dict(class_id=h['box']['class_id'],box_xyxy=[h['box'][k] for k in ('left','top','right','bottom')]) for h in hints]
        row.update(status=result['status'],reason=result['fallback_reason'],primary=len(result['rescue_hints']),additional=len(result['supplementary_hints']),
            metrics=dict(primary=score(adapt(result['rescue_hints']),targets),combined=score(adapt(result['rescue_hints']+result['supplementary_hints']),targets)))
    suite=unittest.TestSuite()
    for name in ('test_consensus_port_rescue.py','test_precision_port_rescue.py','test_bounded_port_rescue.py',
        'test_independent_port_rescue.py','test_port_crop_gui_bridge.py','test_port_state_hint.py','test_port_tiling.py',
        'test_inspection_agent.py','test_inspection_agent_gui_contract.py','test_local_evidence_bridge.py','test_local_review_gui_bridge.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(REPO/'tests'),pattern=name))
    suite.addTests(unittest.TestLoader().discover(str(ROOT/'experiments'),pattern='test_port_rescue_freshness.py'))
    tests=unittest.TextTestRunner(verbosity=1).run(suite);assert tests.wasSuccessful()
    assert {p:sha(Path(p)) for p in protected}==protected
    save(OUT/'report.json',dict(status='complete',tests=tests.testsRun,cases=rows,primary_gui_default_preserved=True,
        extra_default_off=True,actual_gui_runs=4,fresh_initial_runs_without_sam=2,new_sam=False,
        fresh_source_reference_inference=True,post_scoring_recovered_after_directory_read_bug=True,
        recovery_new_inference=False,protected_evidence_sha256=protected,
        caution='Two additional samples are initial DINO geometry only, not complete SAM-fused acceptance.'))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
