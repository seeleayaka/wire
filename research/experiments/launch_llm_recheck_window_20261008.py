"""Opt-in experimental mainline window; overrides worker only in this process."""
import sys,copy
from pathlib import Path

import os
PROJECT=Path(os.environ.get('WIREMIND_PROJECT_ROOT','E:/PythonProject10'))
if not (PROJECT/'prototype').is_dir():
    publication=Path(__file__).resolve().parents[2]
    PROJECT=publication/'mainline' if (publication/'mainline/prototype').is_dir() else publication
sys.path[:0]=[str(PROJECT),str(PROJECT/'models/dinov2'),str(PROJECT/'prototype')]
# Preserve the mainline Windows torch-before-Qt import order.
import assembly_auto_review_dino_v2 as entry
import assembly_auto_review_dino as gui
import llm_recheck_planner as planner
import llm_review_priority as priority
from PyQt5.QtWidgets import QMessageBox, QTableWidget, QTableWidgetItem, QAbstractItemView
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QPushButton,QComboBox,QLabel,QWidget,QVBoxLayout,QCheckBox,QHBoxLayout
from PyQt5.QtCore import QSettings
import private_region_review as private
import deepseek_thinking_options as thinking
from probe_mosaic_decisive_review_20261009 import PHOTO_FIRST

def render_plan(result):
    if result.get('status')!='ok':
        return planner.NOTICE+'\n规划未返回有效结果，请按原本地流程继续复核。'
    plan=result['plan']; by_id={r['candidate_id']:r for r in plan['regions']}
    lines=[planner.NOTICE,plan['summary_zh']]
    for candidate_id in plan['review_order']:
        row=by_id[candidate_id]
        lines += ['',candidate_id+'：'+row['observation_zh'],
                  '建议核查：'+'；'.join(planner.ACTIONS[x] for x in row['requested_checks']),
                  '补证问题：'+row['question_zh']]
    return '\n'.join(lines)

class PlannedWorker(gui.DeepSeekMaskReviewWorker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        parent=kwargs.get('parent')
        self.visual_packet=copy.deepcopy(getattr(parent,'_approved_visual_packet',None))
        self.visual_pending=getattr(parent,'_deepseek_pending',None)
        self.thinking_mode=getattr(parent,'thinking_mode',None).currentData() if getattr(parent,'thinking_mode',None) else 'disabled'
    def run(self):
        try:
            if self.visual_packet is not None:private.validate(self.visual_packet,self.visual_pending)
            diagnostics={}
            settings=thinking.configured_settings(planner.backend.load_settings(),self.thinking_mode)
            sender=thinking.thinking_sender(self.thinking_mode,planner.backend._post_json,
                                           PHOTO_FIRST if self.visual_packet else None,diagnostics)
            result=priority.run(self.reference_mask_path,self.inspection_mask_path,self.candidates,settings=settings,api_key=self.api_key,
                                visual_packet=self.visual_packet,sender=sender)
            result.update(thinking_mode=self.thinking_mode,thinking_diagnostics=diagnostics)
            result.update(answer_zh=render_plan(result),answer_retry_count=0,manual_only=True,
                          external_review_is_non_authoritative=True)
            if result['status']!='ok': result['error']=result.get('safe_error','没有需要规划的候选')
        except Exception:
            result={'status':'error','error':'复核计划未完成，保留原本地结果。','manual_only':True}
        self.completed.emit(result)

class PlannedWindow(gui.DINOReview):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.windowTitle()+' · 大模型结构化复核试验')
        self.deepseek_review_button.setText('大模型判断并调整复核等级')
        self.deepseek_question_input.setText('逐框风险判断、优先级建议与补证步骤')
        self.deepseek_question_input.setReadOnly(True)
        self._approved_visual_packet=None
        self.input_mode=QComboBox()
        self.input_mode.addItem('纯掩膜（默认）','binary_masks')
        self.input_mode.addItem('脱敏局部原图＋掩膜（需预览）','redacted_local_photos')
        self.preview_button=QPushButton('预览并遮挡局部图片（不发送）')
        self.preview_button.setEnabled(False)
        self.preview_button.clicked.connect(self.preview_private_regions)
        self.input_mode.currentIndexChanged.connect(self._input_mode_changed)
        controls=self.result_card.layout()
        self.input_mode_label=QLabel('大模型复核输入')
        controls.addWidget(self.input_mode_label)
        controls.addWidget(self.input_mode);controls.addWidget(self.preview_button)
        self.priority_table=QTableWidget(0,5,self)
        self.priority_table.setHorizontalHeaderLabels(['候选','原等级','模型建议','当前等级','调整依据'])
        self.priority_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.priority_table.setMinimumHeight(150)
        self.priority_table.horizontalHeader().setStretchLastSection(True)
        self.result_card.layout().addWidget(self.priority_table)
        self._attach_cloud_settings()

    def _attach_cloud_settings(self):
        self.cloud_switch=QCheckBox('启用 DeepSeek 辅助复核')
        self.cloud_switch.setToolTip('仅启用设置；不会自动发送图片。')
        self.cloud_body=QWidget();layout=QVBoxLayout(self.cloud_body);layout.setContentsMargins(12,12,12,12);layout.setSpacing(9)
        self.cloud_body.setObjectName('cloudSettings')
        self.cloud_body.setStyleSheet('QWidget#cloudSettings {background:#ffffff;border:1px solid #d9e4de;border-radius:8px;}')
        layout.addWidget(QLabel('DeepSeek 连接'))
        self.deepseek_api_input.parentWidget().layout().removeWidget(self.deepseek_api_input)
        layout.addWidget(self.deepseek_api_input)
        self.deepseek_api_input.setPlaceholderText('已记住密钥，留空使用' if self.deepseek_saved_key_available else '输入 API Key，本机记忆')
        buttons=QHBoxLayout();self.save_key_button=QPushButton('保存密钥');self.save_key_button.clicked.connect(self._save_cloud_key);buttons.addWidget(self.save_key_button)
        hint=QLabel('仅保存在本机，不写入报告或仓库');hint.setWordWrap(True);buttons.addWidget(hint);layout.addLayout(buttons)
        layout.addWidget(QLabel('思考强度'))
        self.thinking_mode=QComboBox()
        for mode,(label,_) in thinking.MODES.items():self.thinking_mode.addItem(label,mode)
        self.cloud_preferences=QSettings('WireMind','CloudReview')
        remembered=str(self.cloud_preferences.value('thinking_mode','low'))
        self.thinking_mode.setCurrentIndex(max(0,self.thinking_mode.findData(remembered)))
        self.thinking_mode.currentIndexChanged.connect(lambda:self.cloud_preferences.setValue('thinking_mode',self.thinking_mode.currentData()))
        layout.addWidget(self.thinking_mode)
        budget=QLabel('轻量思考适合日常复核；标准与深度更耗时。请求超时或无完整回答时保留本地结果。');budget.setWordWrap(True);layout.addWidget(budget)
        for control in (self.deepseek_question_input,self.deepseek_review_button,self.deepseek_answer_label,self.input_mode_label,self.input_mode,self.preview_button,self.priority_table):
            old=control.parentWidget().layout()
            if old:old.removeWidget(control)
            layout.addWidget(control)
        self.deepseek_question_input.hide()
        self.result_card.layout().addWidget(self.cloud_switch);self.result_card.layout().addWidget(self.cloud_body)
        self.cloud_body.hide();self.cloud_switch.toggled.connect(self.cloud_body.setVisible)

    def _save_cloud_key(self):
        token=self.deepseek_api_input.text().strip()
        if not token:self._set_stage('请输入要保存的 API Key。',False);return
        try:planner.backend.save_local_api_key(planner.backend.load_settings(),token)
        except (OSError,ValueError,planner.backend.DeepSeekMaskReviewError):
            self._set_stage('密钥未能保存，请检查本机配置目录权限。',False);return
        self.deepseek_api_input.clear();self.deepseek_saved_key_available=True;self._update_deepseek_api_hint()
        self._set_stage('密钥已记住，仅保存于本机。',False)

    def run(self):
        if self.deepseek_worker is not None and self.deepseek_worker.isRunning():return
        self._approved_visual_packet=None
        self.priority_table.setRowCount(0)
        super().run()

    def _input_mode_changed(self):
        self._approved_visual_packet=None
        self.preview_button.setEnabled(self.input_mode.currentData()=='redacted_local_photos')

    def preview_private_regions(self):
        pending=self._deepseek_pending
        self._approved_visual_packet=None
        if pending is None or self.current_output!=pending['output']:
            self._set_stage('请先完成检测，生成当前复核框。',False);return False
        try:
            prepared=private.prepare(pending)
            dialog=private.PreviewDialog(prepared,self)
            if dialog.exec_()!=dialog.Accepted:return False
            private.validate(dialog.packet,pending)
            self._approved_visual_packet=copy.deepcopy(dialog.packet)
            self._set_stage('本地脱敏预览已确认，尚未发送。',False)
            return True
        except (OSError,ValueError,KeyError,TypeError):
            self._set_stage('局部预览不可用或范围过大，请使用纯掩膜；原检测结果不变。',False)
            return False

    def _finish_deepseek_mask_review(self,result):
        pending=self._deepseek_pending
        if pending is None or self.current_output!=pending['output']:return
        self.input_mode.setEnabled(True)
        self.cloud_switch.setEnabled(True);self.thinking_mode.setEnabled(True);self.save_key_button.setEnabled(True)
        self.preview_button.setEnabled(self.input_mode.currentData()=='redacted_local_photos')
        if result.get('input_mode')=='redacted_local_photos':
            try:
                private.validate(self._approved_visual_packet,pending)
                if result.get('visual_packet_sha256')!=self._approved_visual_packet['packet_sha256'] or result.get('visual_source_binding')!=private.source_binding(pending):
                    raise ValueError('Preview changed')
            except (OSError,ValueError,KeyError,TypeError):
                result={**result,'status':'error','error':'局部图片或预览已变化，保留原本地等级。'}
        try:
            evidence=priority.local_evidence(pending['report'],pending['reference_mask'],pending['inspection_mask'])
            current=priority.binding(pending['reference_mask'],pending['inspection_mask'],pending['candidates'])
            audit=priority.adjust(result,evidence,current)
        except (OSError,ValueError,KeyError,TypeError):
            # Never leave an old priority table visible after a failed computation.
            self.priority_table.setRowCount(0)
            result={**result,'status':'error','error':'本地证据核查未完成，保留原本地等级。'}
            return super()._finish_deepseek_mask_review(result)
        result={**result,'priority_adjustment':audit}
        if result.get('status')=='ok' and not audit['model_result_accepted']:
            result={**result,'status':'error','error':'模型结果与当前输入不符，保留原本地等级。'}
        self.priority_table.setRowCount(len(audit['rows']))
        for i,row in enumerate(audit['rows']):
            values=[row['candidate_id'],priority.LABELS[priority.LEVELS[row['baseline_priority']]],
                    row['llm_suggested_priority'] or '不可用',row['display_label'],row['adjustment_reason_zh']]
            for j,value in enumerate(values):
                cell=QTableWidgetItem(value)
                if j==3:cell.setBackground(QColor({'high':'#fff0dd','normal':'#fff8dc','low':'#edf5ff'}[row['adjusted_priority']]))
                self.priority_table.setItem(i,j,cell)
        self.priority_table.resizeColumnsToContents()
        super()._finish_deepseek_mask_review(result)
        if result.get('status')=='ok':
            level=audit['overall_priority']
            self._set_stage('大模型复核完成 · '+(priority.LABELS[level] if level else '无候选'),False)

    def start_deepseek_mask_review(self):
        if not self.cloud_switch.isChecked():return
        pending=self._deepseek_pending
        if pending is None or self.current_output!=pending['output']:return
        if any(w is not None and w.isRunning() for w in [self.deepseek_worker,self.port_worker,getattr(self,'rescue_worker',None)]):return
        raw=self.input_mode.currentData()=='redacted_local_photos'
        if raw:
            if self._approved_visual_packet is None and not self.preview_private_regions():return
            try:private.validate(self._approved_visual_packet,pending)
            except (OSError,ValueError,KeyError,TypeError):
                self._approved_visual_packet=None
                self._set_stage('输入已变化，请重新预览局部图片。',False);return
            scope='将本次已预览并遮挡的局部参考图、局部待检图及对应掩膜发送至配置的DeepSeek服务。\n不发全柜照片或原有文字编号，但线色、接头和局部布线仍可见，自动遮挡不保证无遗漏。'
        else:
            self._approved_visual_packet=None
            scope='将本次两张二值布线掩膜、候选坐标和最少差异信息发送至配置的DeepSeek服务。\n不发送原照片，但掩膜仍可能暴露布线布局。'
        answer=QMessageBox.question(self,'确认本次复核外发',scope+
            '\n判断只影响复核优先级，不删除框、不修改连接结论。是否继续？',
            QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if answer==QMessageBox.Yes:
            super().start_deepseek_mask_review()
            if self.deepseek_worker is not None:
                self.input_mode.setEnabled(False);self.preview_button.setEnabled(False)
                self.cloud_switch.setEnabled(False);self.thinking_mode.setEnabled(False);self.save_key_button.setEnabled(False)

def main():
    gui.DeepSeekMaskReviewWorker=PlannedWorker
    gui.DINOReview=PlannedWindow
    entry.main()

if __name__=='__main__':main()
