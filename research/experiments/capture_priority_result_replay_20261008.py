"""Capture real Qt widgets replaying saved results, no API/SAM/project mutation."""
import copy,json,hashlib
from pathlib import Path
import launch_llm_recheck_window_20261008 as ui
from PyQt5.QtWidgets import QApplication
from probe_llm_recheck_planner_20261008 import ROOT,CASES

def main():
    app=QApplication([]);app.setStyle('Fusion')
    out=Path('C:/Users/HUAWEI/Pictures/Screenshots');out.mkdir(exist_ok=True)
    jobs=[('cabinet2',ROOT/'artifacts/llm_priority_cabinet2_diagnostic_20261008/result.json','机柜2 · 本次诊断回放 · 云端8.25秒'),
          ('cabinet4_right1',ROOT/'artifacts/llm_review_priority_20261008/cabinet4_right1.json','机柜4 right1 · 上轮实测回放 · 云端8.50秒')]
    receipt=[]
    for name,result_path,title in jobs:
        source=CASES[name];before=hashlib.sha256(source.read_bytes()).hexdigest()
        report=json.loads(source.read_text(encoding='utf-8'));result=json.loads(result_path.read_text(encoding='utf-8'))
        original=copy.deepcopy(report);fusion=report['sam3_fusion'];folder=source.parent
        ref=Path(fusion['reference_sam3']['output_dir'])/'mask_union.png'
        ins=Path(fusion['inspection_sam3']['output_dir'])/'mask_union.png'
        w=ui.PlannedWindow();w.setWindowTitle(title+'｜实测结果回放，不是重新推理')
        w.reference.setText(report['reference'])
        w.inspection.setText(report['inspection'])
        saved=[];w._write_report=lambda value:saved.append(copy.deepcopy(value))
        w.current_output=folder
        w._deepseek_pending={'output':folder,'report':report,'reference_mask':ref,'inspection_mask':ins,'candidates':report['review_regions']}
        w.aligned_view.load(folder/'aligned.jpg')
        w.dino_boxes_view.load(folder/'dino_anomaly_boxes.jpg')
        w.sam_difference_view.load(folder/'sam3_difference_boxes.jpg')
        w.sam_detection_view.load(folder/'sam3/inspection/overlay.jpg')
        w.fusion_view.load(folder/'sam3_fusion_boxes.jpg')
        w.tabs.setCurrentWidget(w.fusion_view)
        w.set_decision('uncertain',title+'｜'+str(len(report['review_regions']))+'个候选均保留','回放已保存的视觉结果与真实大模型回复；未重跑SAM，未发起新的API请求。')
        result['answer_zh']=ui.render_plan(result)
        w._finish_deepseek_mask_review(result)
        w.resize(1660,1200);w.show();app.processEvents()
        for view in [w.fusion_view,w.aligned_view,w.dino_boxes_view,w.sam_difference_view,w.sam_detection_view]:view.fit_image()
        app.processEvents()
        path=out/f'WireMind_{name}_priority_result_replay_final_20261008.png'
        if path.exists():raise FileExistsError('Preserve prior screenshot')
        assert w.grab().save(str(path),'PNG')
        # Capture the actual answer widget separately; the optional panel is collapsed by default.
        w.workbench_optional_toggle.click();app.processEvents()
        answer_path=out/f'WireMind_{name}_model_answer_replay_20261008.png'
        if answer_path.exists():raise FileExistsError('Preserve prior answer screenshot')
        assert w.deepseek_answer_label.grab().save(str(answer_path),'PNG')
        assert report['decision']==original['decision'] and report['review_regions']==original['review_regions']
        assert before==hashlib.sha256(source.read_bytes()).hexdigest()
        receipt.append({'image':str(path),'answer_image':str(answer_path),'source_report_sha256':before,'result_sha256':hashlib.sha256(result_path.read_bytes()).hexdigest(),
                        'screenshot_type':'actual_Qt_widget_capture_of_saved_result_replay','new_api_calls':0,'sam_rerun':False,'source_unchanged':True})
        w.close();w.deleteLater();app.processEvents()
    (ROOT/'artifacts/llm_priority_result_screenshot_receipt_20261008.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
