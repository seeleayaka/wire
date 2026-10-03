"""Post-inference audit and optional-port admission check on real saved reports."""
import os
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/main_scenario_live_20261002'
REPO=Path('E:/PythonProject10')
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'prototype'))
os.environ.update(YOLO_CONFIG_DIR=str(OUT/'port_config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False')
import torch
from inspection_agent import InspectionTask
from inspection_agent.port_crop_gui_bridge import run_gui_port_review,render_port_overlay
from inspection_agent.optional_port_crop_review import SCENE

def main():
    # Preserve the first environment failure; only the harness config path changes.
    previous=OUT/'acceptance_audit.json'
    first=OUT/'acceptance_audit_environment_first_attempt.json'
    if previous.exists() and not first.exists():
        first.write_bytes(previous.read_bytes())
    (OUT/'port_config/Ultralytics').mkdir(parents=True,exist_ok=True)
    report=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
    rows=[]
    for row in report['cases']:
        if row['status']=='error':continue
        output=Path(row['output']);task_path=Path(row['task'])
        before=hashlib.sha256(task_path.read_bytes()).hexdigest()
        task=InspectionTask.load(task_path).to_report()
        visual=json.loads((output/'report.json').read_text(encoding='utf-8'))
        assert task['state']=='awaiting_human_review'
        assert task['machine_evidence'][0]['candidate_count']==len(visual.get('review_regions',[]))
        assert not task['human_conclusions'] and not task['repair_guidance'] and not task['topology_assessments']
        optional=run_gui_port_review(visual,project=REPO,enabled=True,scene=SCENE)
        (output/'optional_port_admission_audit.json').write_text(json.dumps(optional,ensure_ascii=False,indent=2),encoding='utf-8')
        if optional['status']=='applied':render_port_overlay(output/'aligned.jpg',optional,output/'optional_port_overlay.jpg')
        assert before==hashlib.sha256(task_path.read_bytes()).hexdigest()
        counts=row['evaluation']['stages']['fusion']
        rows.append(dict(image=row['image'],task_state=task['state'],sam_status=row['sam_status'],dino_candidates=row['dino_candidates'],fusion_candidates=row['fusion_candidates'],target_fragments=counts['target_count'],fragment_any_overlap=counts['target_any_overlap'],fragment_iou025=counts['target_iou025'],fragment_iou050=counts['target_iou050'],optional_port_status=optional['status'],optional_port_reason=optional['fallback_reason'],new_port_hints=len(optional['tile_hints']),human_conclusions=0,reinspection_performed=False))
    assert len(rows)==4
    result={'cases':rows,'fresh_real_inference':True,'field_accuracy_claimed':False,'human_confirmation':False,'new_repair_image_available':False,'reinspection_performed':False,'task_bytes_unchanged':True}
    (OUT/'acceptance_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 正式入口四图真实推理核查 / 2026-10-02','',
       '固定参考 train01/normal_073.JPG；test01 每种文件名前缀各取排序第一张。该批已用于历史实验，不是新未见测试集。全部沿用现有模型和参数，不调整候选。当前正式入口 LARGE_ROI_MAX_CANDIDATES=3，不能直接套用历史预算6/CNN实验的成绩。',
       '', '| 样例 | DINO框 | 融合框 | 源标签片段重叠 | IoU≥0.25 | IoU≥0.5 | 端口入口 |','|---|---:|---:|---:|---:|---:|---|']
    for r in rows:lines.append(f"| {r['image']} | {r['dino_candidates']} | {r['fusion_candidates']} | {r['fragment_any_overlap']}/{r['target_fragments']} | {r['fragment_iou025']} | {r['fragment_iou050']} | {r['optional_port_status']}: {r['optional_port_reason']} |")
    lines+=['','## 边界','',
       '零标签正常图中的候选是额外复核负担；故障图上的任意相交只是弱定位命中，不是准确故障识别。标签是源数据片段框，不是每一条独立故障或电气连通真值；局部校正后的候选与仅经全局H转换的标签可能有局部残差。查看 evaluation_overlay.jpg：青色T为源标签、橙色C为融合候选。',
       '', '工单已从实际结果创建并保存重载，仍 awaiting_human_review；禁止未确认维修建议的检查通过。未填写真实人工意见，没有真实维修后照片，没有执行复检，不能宣称完整维修闭环通过。',
       '', '端口检查是独立的可选入口准入核查，不改变本轮基线结果。若局部ECC触发 local_alignment_not_supported，代表可选模型未执行，不能把旧端口定位提升计入当前默认主链。SAM新推理/缓存状态见各 report.json；本轮参考缓存只在同一批的后续样例复用。','',
       '下一步先处理实测暴露的主链问题，在相同固定评估批上对照，不针对这四张调参。保留人审门槛；拓扑身份/预期表缺口仍在。']
    (OUT/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
