"""Display this run's completed reports with local fonts; never start workers."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/fresh_four_examples_20261004'
REPO = Path('E:/PythonProject10')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.dont_write_bytecode = True
sys.path[:0] = [str(REPO), str(REPO / 'prototype')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    import assembly_auto_review_dino_v2 as entry
    from PyQt5.QtGui import QFontDatabase, QFont
    from PyQt5.QtWidgets import QPushButton
    gui = entry.implementation
    app = gui.QApplication([])
    app.setStyle('Fusion')
    font_id = QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
    assert QFontDatabase.applicationFontFamilies(font_id)
    app.setFont(QFont('Microsoft YaHei', 10))
    rows = json.loads((OUT / 'progress.json').read_text(encoding='utf-8'))['cases']
    rendered = []
    for row in rows:
        output = Path(row['output'])
        report_path = output / 'report.json'
        task_path = Path(row['task'])
        before = {str(p): digest(p) for p in (report_path, task_path)}
        report = json.loads(report_path.read_text(encoding='utf-8'))
        window = gui.DINOReview()
        window.resize(1400, 950)
        window.reference.setText(report['reference'])
        window.inspection.setText(report['inspection'])
        window.current_output = output
        window.agent_task_path = task_path
        window._set_agent_state('awaiting_human_review', task_path)
        window._set_stage('本轮检测结果（读取本轮报告，不重新推理）', False)
        window.set_decision('uncertain', '待人工复核',
                            f"差异区域 {row['fusion_cues']} 个；端口主要 {row['main_ports']} 个、补充 {row['supplementary_ports']} 个。")
        for view, name in ((window.aligned_view, 'aligned.jpg'),
                           (window.dino_boxes_view, 'dino_anomaly_boxes.jpg'),
                           (window.dino_heat_view, 'check_heatmap.jpg'),
                           (window.fusion_view, 'sam3_fusion_boxes.jpg')):
            if (output / name).is_file():
                view.load(output / name)
        window.tabs.setCurrentWidget(window.fusion_view)
        if row.get('port_overlay'):
            window.rescue_view.load(Path(row['port_overlay']))
            window.tabs.setCurrentWidget(window.rescue_view)
            window.rescue_status.setText(f"本轮端口提示：主要 {row['main_ports']} 个，补充 {row['supplementary_ports']} 个；仍需人工复核。")
        elif row['ports_status'] == 'fallback':
            window.rescue_status.setText('端口分支退出：暂不支持本轮局部对齐变换；不是正常识别成功。')
        for button in window.findChildren(QPushButton):
            button.setEnabled(False)
        window.show()
        app.processEvents()
        target = OUT / Path(row['image']).stem / 'desktop_result_fonts_loaded.png'
        assert window.grab().save(str(target))
        assert window.initial_worker is None and window.sam3_worker is None and window.rescue_worker is None
        window.close()
        app.processEvents()
        assert all(digest(Path(p)) == value for p, value in before.items())
        rendered.append(dict(image=row['image'],screenshot=str(target),reports_unchanged=True))
    result = dict(status='complete',purpose='display_this_run_only',model_inference=False,
                  original_desktop_grabs_preserved=True,rendered=rendered)
    (OUT / 'desktop_display_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
