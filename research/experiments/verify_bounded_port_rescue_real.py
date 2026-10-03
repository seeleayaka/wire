import sys,json,os
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,'E:/PythonProject10')
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/bounded_port_rescue_real_20261002'
os.environ.update(YOLO_CONFIG_DIR=str(OUT/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False')
from inspection_agent.bounded_port_rescue import run_bounded_port_rescue
from inspection_agent.optional_port_crop_review import SCENE,sha
from inspection_agent.port_crop_gui_bridge import render_port_overlay
def main():
    if OUT.exists():raise FileExistsError('Fresh output required')
    (OUT/'config/Ultralytics').mkdir(parents=True)
    live=ROOT/'artifacts/rescue_mainline_ab_20261002_v2'
    cases=json.loads((live/'report.json').read_text(encoding='utf-8'))['cases'];rows=[]
    for case in cases:
        output=Path(case['output']);file=output/'report_off.json';report=json.loads(file.read_text(encoding='utf-8'))
        digest=sha(file);result=run_bounded_port_rescue(report,project='E:/PythonProject10',enabled=True,scene=SCENE)
        target=OUT/Path(case['image']).stem;target.mkdir()
        (target/'evidence.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        assert result['parents']==report['review_regions'] and sha(file)==digest
        if result['status']=='applied':render_port_overlay(output/'aligned.jpg',dict(parents=result['parents'],tile_hints=result['rescue_hints']),target/'overlay.jpg')
        rows.append(dict(image=case['image'],status=result['status'],hints=len(result['rescue_hints']),reason=result['fallback_reason']))
        print(json.dumps(rows[-1]),flush=True)
    (OUT/'report.json').write_text(json.dumps(dict(cases=rows,new_source_and_reference_inference=True,new_dino_sam=False,
        baseline_regions_unchanged=True,gui_budget_one_unchanged=True),indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
