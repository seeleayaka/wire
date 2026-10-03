"""Independent optional GUI entry with request/report/image freshness binding."""
import copy
import hashlib
import json
import os
import uuid
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QCheckBox, QLabel, QPushButton
from inspection_agent.independent_port_rescue import review_rois
from inspection_agent.context_port_recheck import run_context_port_recheck as run_consensus_port_rescue, render_consensus_overlay, POLICY_ID
from inspection_agent.optional_port_crop_review import sha
from inspection_agent.teacher_student_port_support import (run_teacher_student_review,
    support_runtime_fingerprint, POLICY_ID as STUDENT_POLICY_ID)
from inspection_agent.paired_native_pose import (run_native_pose_review as run_feature_residual_review,
    native_pose_runtime_fingerprint as residual_runtime_fingerprint, POLICY_ID as FEATURE_POLICY_ID)


def report_digest(report):
    return hashlib.sha256(json.dumps(report, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def snapshot(window):
    report = window._port_visual_report
    output = Path(window.current_output)
    supplementary=bool(getattr(window,'rescue_supplement_switch',None)
        and window.rescue_supplement_switch.isChecked())
    requested=bool(getattr(window,'rescue_student_switch',None) and window.rescue_student_switch.isChecked())
    student=bool(supplementary and requested)
    feature_requested=bool(getattr(window,'rescue_feature_switch',None) and window.rescue_feature_switch.isChecked())
    feature=bool(student and feature_requested)
    return dict(report=report_digest(report), disk=sha(output / 'report.json'), aligned=sha(output / 'aligned.jpg'),
                output=str(output), source=sha(report['inspection']), reference=sha(report['reference']),
                input_source=window.inspection.text().strip(), input_reference=window.reference.text().strip(),
                scene=str(window.port_scene_combo.currentData()), policy=FEATURE_POLICY_ID if feature else STUDENT_POLICY_ID if student else POLICY_ID,
                supplementary=supplementary, student=student, student_requested=requested,
                student_runtime=support_runtime_fingerprint(Path(__file__).resolve().parents[1]) if student else None,
                feature=feature,feature_requested=feature_requested,
                feature_runtime=residual_runtime_fingerprint(Path(__file__).resolve().parents[1]) if feature else None)


def payload_is_current(window, payload):
    try:
        return (window.rescue_switch.isChecked() and payload['token'] == window._rescue_token
                and payload['snapshot'] == snapshot(window))
    except (OSError, TypeError, ValueError, KeyError):
        return False


class RescueWorker(QThread):
    completed = pyqtSignal(object)

    def __init__(self, report, output, token, binding, parent=None):
        super().__init__(parent)
        self.report_snapshot = copy.deepcopy(report)
        self.output = Path(output)
        self.token, self.binding = token, binding

    def run(self):
        evidence = self.output / ('port_rescue_' + self.token)
        evidence.mkdir(parents=True, exist_ok=True)
        try:
            config = evidence / 'model_config'
            (config / 'Ultralytics').mkdir(parents=True, exist_ok=True)
            os.environ['YOLO_CONFIG_DIR'] = str(config)
            if self.binding.get('feature',False):
                result = run_feature_residual_review(self.report_snapshot,
                    project=Path(__file__).resolve().parents[1], enabled=True, scene=self.binding['scene'],
                    supplementary_enabled=self.binding['supplementary'],student_enabled=True,feature_enabled=True,resolution_enabled=True,paired_enabled=True,median_enabled=True,native_pose_enabled=True)
            elif self.binding.get('student',False):
                result = run_teacher_student_review(self.report_snapshot,
                    project=Path(__file__).resolve().parents[1], enabled=True, scene=self.binding['scene'],
                    supplementary_enabled=self.binding['supplementary'],student_enabled=True)
            else:
                result = run_consensus_port_rescue(self.report_snapshot,
                    project=Path(__file__).resolve().parents[1], enabled=True, scene=self.binding['scene'],
                    supplementary_enabled=self.binding['supplementary'])
            if result['status'] == 'applied':
                render_consensus_overlay(self.output / 'aligned.jpg',result,evidence / 'overlay.jpg')
            (evidence / 'evidence.json').write_text(json.dumps(dict(result=result, binding=self.binding),
                ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as error:
            result = dict(status='fallback', fallback_reason=str(error), rescue_hints=[], automatic_fault_verdict=False)
        self.completed.emit(dict(result=result, token=self.token, snapshot=self.binding, evidence=str(evidence)))


def attach_port_rescue(window, image_view_class):
    window.rescue_worker = None
    window._rescue_token = None
    window.rescue_view = image_view_class('独立端口补漏默认关闭')
    window.tabs.addTab(window.rescue_view, '独立端口补漏')
    window.rescue_switch = QCheckBox('启用独立端口补漏（最多5个高分线索）', window)
    window.rescue_supplement_switch = QCheckBox('附加高可靠线索（含局部重查，另最多5框）', window)
    window.rescue_student_switch = QCheckBox('新模型补强（保留原检测，共用附加5框额度）', window)
    window.rescue_feature_switch = QCheckBox('增强补漏（实验，保留现有结果）', window)
    window.rescue_feature_switch.setToolTip('保留已有结果，增加经过正常参考与定位核查的线索；共用原附加额度，仍需人工复核。')
    window.rescue_button = QPushButton('检查端口补漏', window)
    window.rescue_status = QLabel('补漏：关闭；只提供人工复核线索，不确认断接。', window)
    window.rescue_status.setWordWrap(True)
    for control in (window.rescue_switch, window.rescue_supplement_switch, window.rescue_student_switch, window.rescue_feature_switch, window.rescue_button, window.rescue_status):
        window.workbench_optional_body.layout().addWidget(control)

    def invalidate(*_):
        window._rescue_token = None
        window.rescue_view.scene.clear()
        window.rescue_view.item = None
        window.rescue_status.setText('补漏：尚未运行；当前模型仅支持已确认的现有机箱场景。')
        update_port_rescue_controls(window)

    def start():
        update_port_rescue_controls(window)
        if not window.rescue_button.isEnabled():
            return
        try:
            binding = snapshot(window)
        except (OSError, TypeError, ValueError):
            window.rescue_status.setText('补漏：输入或报告不可读，请重新视觉复核。')
            return
        token = uuid.uuid4().hex
        window._rescue_token = token
        worker = RescueWorker(window._port_visual_report, window.current_output, token, binding, window)
        window.rescue_worker = worker
        window.rescue_status.setText('补漏：正在检查原图与参考图；保留原候选。')
        worker.completed.connect(lambda payload: finish_port_rescue(window, payload))
        worker.finished.connect(worker.deleteLater)
        window.run_button.setEnabled(False)
        window.deepseek_review_button.setEnabled(False)
        worker.start()
        window._update_port_controls()

    window.rescue_switch.toggled.connect(invalidate)
    window.rescue_supplement_switch.toggled.connect(invalidate)
    window.rescue_student_switch.toggled.connect(invalidate)
    window.rescue_feature_switch.toggled.connect(invalidate)
    window.reference.textChanged.connect(invalidate)
    window.inspection.textChanged.connect(invalidate)
    window.port_scene_combo.currentIndexChanged.connect(invalidate)
    window.rescue_button.clicked.connect(start)
    update_port_rescue_controls(window)


def update_port_rescue_controls(window):
    busy = any(getattr(window, name, None) is not None and getattr(window, name).isRunning()
               for name in ('initial_worker','sam3_worker','deepseek_worker','port_worker','rescue_worker'))
    ready = window._port_visual_report is not None and window.current_output is not None
    try:
        if ready:
            review_rois(window._port_visual_report)
            ready = (window._port_visual_report.get('decision') in {
                'possible_difference_manual_review','no_significant_wire_related_difference','no_significant_difference'}
                and Path(window.reference.text().strip()) == Path(window._port_visual_report['reference'])
                and Path(window.inspection.text().strip()) == Path(window._port_visual_report['inspection']))
    except (ValueError, TypeError, KeyError):
        ready = False
    from inspection_agent.optional_port_crop_review import SCENE
    window.rescue_button.setEnabled(bool(ready and not busy and window.rescue_switch.isChecked()
                                        and window.port_scene_combo.currentData() == SCENE))
    window.rescue_switch.setEnabled(not busy)
    window.rescue_supplement_switch.setEnabled(not busy)
    if getattr(window,'rescue_student_switch',None):
        window.rescue_student_switch.setEnabled(not busy and window.rescue_supplement_switch.isChecked())
    if getattr(window,'rescue_feature_switch',None):
        window.rescue_feature_switch.setEnabled(not busy and window.rescue_supplement_switch.isChecked()
            and window.rescue_student_switch.isChecked())


def finish_port_rescue(window, payload):
    # A stale completion cannot discard another request's worker or rewrite reports.
    if not payload_is_current(window, payload):
        if window.rescue_worker is not None and window.rescue_worker.token == payload['token']:
            window.rescue_worker = None
            window.rescue_status.setText('补漏：输入或报告已变化，旧结果未回填。')
        window.run_button.setEnabled(window.rescue_worker is None)
        window.deepseek_review_button.setEnabled(window.rescue_worker is None and window._deepseek_pending is not None)
        window._update_port_controls()
        return
    window.rescue_worker = None
    result = payload['result']
    report = copy.deepcopy(window._port_visual_report)
    report['independent_port_rescue'] = dict(status=result['status'],
        rescue_hints=result['rescue_hints'], automatic_fault_verdict=False,
        fallback_reason=result.get('fallback_reason'), evidence_path=payload['evidence'],
        request_binding=payload['snapshot'], audit=result.get('audit'), budget_policy=result.get('budget_policy'),
        supplementary_hints=result.get('supplementary_hints',[]), supplementary_policy=result.get('supplementary_policy'),
        recheck_policy=result.get('recheck_policy'),teacher_student_policy=result.get('teacher_student_policy'),
        feature_residual_policy=result.get('feature_residual_policy'),resolution_policy=result.get('resolution_policy'),
        paired_geometry_policy=result.get('paired_geometry_policy'),median_geometry_policy=result.get('median_geometry_policy'),
        native_pose_policy=result.get('native_pose_policy'))
    window._write_report(report)
    if result['status'] == 'applied':
        window.rescue_view.load(Path(payload['evidence']) / 'overlay.jpg')
        window.tabs.setCurrentWidget(window.rescue_view)
        window.rescue_status.setText(f"补漏：{len(result['rescue_hints'])} 个主要线索，{len(result.get('supplementary_hints',[]))} 个附加线索；需人工复核。证据：{payload['evidence']}")
        optional_failure=(result.get('native_pose_policy') or {}).get('fallback_reason') or (result.get('median_geometry_policy') or {}).get('fallback_reason') or (result.get('paired_geometry_policy') or {}).get('fallback_reason') or (result.get('resolution_policy') or {}).get('fallback_reason') or (result.get('feature_residual_policy') or {}).get('fallback_reason') or (result.get('teacher_student_policy') or {}).get('fallback_reason')
        if optional_failure:
            window.rescue_status.setText(window.rescue_status.text()+'；增强未应用，保留旧检测：'+str(optional_failure))
    else:
        window.rescue_status.setText('补漏：保留原结果；' + str(result.get('fallback_reason')))
    window.run_button.setEnabled(True)
    window.deepseek_review_button.setEnabled(window._deepseek_pending is not None)
    window._update_port_controls()
