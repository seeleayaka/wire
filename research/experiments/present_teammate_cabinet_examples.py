"""Capture actual Qt result widgets for this fresh run only, with readable fonts."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/teammate_cabinet_examples_fresh_20261008'
REPO=Path('E:/PythonProject10');sys.dont_write_bytecode=True
os.environ['QT_QPA_PLATFORM']='offscreen';sys.path[:0]=[str(REPO),str(REPO/'prototype')]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    import assembly_auto_review_dino_v2 as entry
    from PyQt5.QtGui import QFontDatabase,QFont
    from PyQt5.QtWidgets import QPushButton
    from PIL import Image,ImageDraw,ImageFont
    gui=entry.implementation;app=gui.QApplication([]);app.setStyle('Fusion')
    fid=QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc');assert QFontDatabase.applicationFontFamilies(fid)
    app.setFont(QFont('Microsoft YaHei',10));share=OUT/'share_screenshots';share.mkdir(exist_ok=True)
    state=json.loads((OUT/'progress.json').read_text(encoding='utf-8'));rendered=[]
    for row in state['cases']:
        output=Path(row['output']);report_path=output/'report.json';task_path=Path(row['task']);before={str(p):sha(p) for p in [report_path,task_path]}
        report=json.loads(report_path.read_text(encoding='utf-8'));assert row['fresh_reference_SAM'] and row['fresh_inspection_SAM']
        assert row['ports_status']=='not_run' and not row['human_confirmation']
        window=gui.DINOReview();window.resize(1440,980)
        window.reference.setText(row['reference']);window.inspection.setText(row['source']);window.current_output=output
        window.agent_task_path=task_path;window._set_agent_state('awaiting_human_review',task_path)
        window._set_stage('本次新运行结果 · 待人工复核',False)
        window.set_decision('uncertain','待人工复核',f"本次外观差异候选 {row['fusion_cues']} 个；不是电气故障确认。")
        for view,name in [(window.aligned_view,'aligned.jpg'),(window.dino_boxes_view,'dino_anomaly_boxes.jpg'),
                (window.dino_heat_view,'check_heatmap.jpg'),(window.fusion_view,'sam3_fusion_boxes.jpg')]:
            if (output/name).is_file():view.load(output/name)
        window.tabs.setCurrentWidget(window.fusion_view)
        for button in window.findChildren(QPushButton):button.setEnabled(False)
        window.show();app.processEvents();screenshot=share/(row['id']+'_程序截图.png');assert window.grab().save(str(screenshot))
        window.close();app.processEvents();assert all(sha(p)==v for p,v in before.items())
        assert window.initial_worker is None and window.sam3_worker is None
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25);canvas=Image.new('RGB',(1740,1160),'#f6f7f9');draw=ImageDraw.Draw(canvas)
        draw.text((20,12),row['id']+' · 本次从原图重新运行',font=font,fill='#20252b')
        panels=[('标准原图',Path(row['reference'])),('待检原图',Path(row['source'])),('本次检测候选',output/'sam3_fusion_boxes.jpg')]
        for i,(title,path) in enumerate(panels):
            x=20+i*580;draw.text((x,62),title,font=font,fill='#20252b')
            image=Image.open(path).convert('RGB');image.thumbnail((550,970));canvas.paste(image,(x+(550-image.width)//2,105+(970-image.height)//2))
        draw.text((20,1100),f"{row['fusion_cues']} 个待复核区域 | 实际耗时 {row['seconds']/60:.1f} 分钟 | 合成变体演示，不是现场电气准确率",font=font,fill='#20252b')
        comparison=share/(row['id']+'_原图对照.png');canvas.save(comparison)
        rendered.append(dict(id=row['id'],screenshot=str(screenshot),comparison=str(comparison),fusion_cues=row['fusion_cues'],seconds=row['seconds']))
    audit=dict(status='complete' if state['status']=='complete' else 'partial',rendered=rendered,
        screenshot_kind='actual_Qt_widget_current_run_report_display',new_inference_in_renderer=False,
        original_live_window_grabs_retained=True,reports_unchanged=True,human_confirmation=False)
    (share/'截图说明.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 线材柜子：本次新运行截图','',
        '正式程序从原始照片重新配准、DINO分析、参考/待检SAM分割后生成。没有复用旧中间结果。',
        '程序截图是实际Qt结果窗口加载本次报告后的抓图；没有再运行检测、改候选或代填人工确认。',
        '框表示外观差异候选，不是已确认电气故障。照片包含生成变体，只供功能演示，不是现场准确率评测。','']
    for r in rendered:lines.append(f"- {r['id']}：{r['fusion_cues']} 个待复核区域，{r['seconds']/60:.1f} 分钟。")
    (share/'给队友看的说明.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
