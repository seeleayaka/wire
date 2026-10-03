"""V3 must retain every v1 source cue; cache/fingerprint verification explicit."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
V1=ROOT/'artifacts/port_resolution_loose_plug_20261003'
V3=ROOT/'artifacts/resolution_plug_final_budget_20261003'
OUT=ROOT/'artifacts/resolution_plug_policy_parity_20261003'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def canonical(rows):return [(r['class_id'],r['box_xyxy'],r['confidence']) for r in rows]
def main():
    if OUT.exists():raise FileExistsError('Preserve evidence')
    from inspection_agent.optional_port_crop_review import sha
    verified=0;identical=0;gained_cases=[];pins={}
    for stage in ('train','inner','outer'):
        for row in load(V1/stage/'report.json')['cases']:
            name=row['image'];filename=Path(name).stem+'_predictions.json'
            p,q=V1/stage/filename,V3/stage/filename;pins[str(p)],pins[str(q)]=sha(p),sha(q)
            a,b=load(p)['trial'],load(q)['trial'];old,new=canonical(a['all_predictions']),canonical(b['all_predictions'])
            assert new[:len(old)]==old,(stage,name,'lost/reordered v1 cue')
            identical+=old==new
            if old!=new:gained_cases.append(dict(stage=stage,image=name))
            verified+=1
    assert verified==270 and {p:sha(Path(p)) for p in pins}==pins
    OUT.mkdir();(OUT/'report.json').write_text(json.dumps(dict(status='complete',verified=verified,
        identical=identical,gained_cases=gained_cases,v1_cue_prefix_preserved=True,pins=pins,
        metadata_policy_id_intentionally_changes=True,new_model_inference=False,field_accuracy=False),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(verified=verified,identical=identical,gained_cases=gained_cases)),flush=True)
if __name__=='__main__':main()
