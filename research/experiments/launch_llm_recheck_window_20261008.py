"""Opt-in experimental mainline window; overrides worker only in this process."""
import sys,copy,time
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
from PyQt5.QtGui import QColor,QPixmap,QFont,QPainter,QPen
from PyQt5.QtWidgets import QPushButton,QComboBox,QLabel,QWidget,QVBoxLayout,QCheckBox,QHBoxLayout,QScrollArea,QSizePolicy,QDialog,QSplitter,QTextBrowser
from PyQt5.QtCore import QSettings,QTimer,pyqtSignal,Qt,QThread
import private_region_review as private
import deepseek_thinking_options as thinking
from llm_visual_review_policy import PHOTO_FIRST

def probe_model_connection(settings,api_key=None,transport=None):
    """Tiny text-only completion; never transmits visual evidence or user metadata."""
    import json,socket
    from urllib.request import Request
    from urllib.error import HTTPError,URLError
    start=time.monotonic()
    try:
        token=planner.backend._api_token(settings,api_key)
    except Exception:
        return {'ok':False,'message':'尚未配置可用密钥，请输入或保存密钥。'}
    try:
        body={'model':settings.model,'messages':[{'role':'user','content':'Reply OK.'}],
              'max_tokens':16,'stream':False,'thinking':{'type':'disabled'}}
        req=Request(settings.endpoint,data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+token},method='POST')
        response=(transport or planner.backend._post_json)(req,20)
        content=response.get('choices',[{}])[0].get('message',{}).get('content')
        if not isinstance(content,str) or not content.strip():return {'ok':False,'message':'服务已响应，但没有有效模型回答，请核对模型配置。'}
        return {'ok':True,'message':'模型连接成功 · '+str(round(time.monotonic()-start,1))+' 秒（仅文字请求）'}
    except Exception as error:
        # Never expose exception bodies, URLs or tokens in UI/report.
        chain=[error,getattr(error,'__cause__',None)]
        code=next((e.code for e in chain if isinstance(e,HTTPError)),None)
        if code in (401,403):message='密钥无效或没有访问权限，请检查密钥。'
        elif code==404:message='接口或模型不可用，请检查配置。'
        elif code==429:message='服务限流或账户额度不足，请稍后重试。'
        elif any(isinstance(e,(TimeoutError,socket.timeout)) for e in chain):message='连接超时，请检查网络或服务状态。'
        elif isinstance(error,planner.backend.DeepSeekMaskReviewError) and 'API key' in str(error):message='尚未配置可用密钥，请输入或保存密钥。'
        else:message='模型连接失败，请检查网络、接口及模型配置。'
        return {'ok':False,'message':message}

class ConnectionProbeWorker(QThread):
    completed=pyqtSignal(dict)
    def __init__(self,settings,api_key,parent=None):
        super().__init__(parent);self.settings=settings;self.api_key=api_key
    def run(self):
        try:self.completed.emit(probe_model_connection(self.settings,self.api_key))
        finally:self.api_key=None

class ReviewReadingDialog(QDialog):
    """Local-only immutable result snapshot; map regions by ID, never by order."""
    def __init__(self,result,images,parent=None):
        super().__init__(parent)
        self.result=copy.deepcopy(result);self.images=images
        self.setWindowTitle('DeepSeek 复核意见与局部图片');self.resize(1200,820)
        layout=QVBoxLayout(self);toolbar=QHBoxLayout()
        toolbar.addWidget(QLabel('复核框'))
        self.selector=QComboBox();self.selector.addItem('总体意见',None)
        self.rows={r['candidate_id']:r for r in result.get('plan',{}).get('regions',[])}
        for cid in result.get('plan',{}).get('review_order',[]):
            if cid in self.rows:self.selector.addItem(cid,cid)
        toolbar.addWidget(self.selector,1);toolbar.addWidget(QLabel('字号'))
        self.font_size=QComboBox()
        for size in [14,16,18,22,26]:self.font_size.addItem(str(size),size)
        self.font_size.setCurrentIndex(1);toolbar.addWidget(self.font_size)
        layout.addLayout(toolbar)
        note=QLabel('右侧为对应复核框的本机原图对照，仅用于本地阅读，不会再次发送。模型意见是辅助复核建议。')
        note.setWordWrap(True);note.setStyleSheet('color:#667785;padding:6px;');layout.addWidget(note)
        split=QSplitter(Qt.Horizontal);layout.addWidget(split,1)
        self.opinion=QTextBrowser();self.opinion.setOpenExternalLinks(False);split.addWidget(self.opinion)
        self.picture_scroll=QScrollArea();self.picture_scroll.setWidgetResizable(True)
        body=QWidget();self.picture_layout=QVBoxLayout(body)
        self.picture_labels=[]
        for title in ['参考局部图','待检局部图（对齐后）']:
            self.picture_layout.addWidget(QLabel(title))
            label=QLabel();label.setAlignment(Qt.AlignCenter);label.setMinimumSize(100,100)
            self.picture_layout.addWidget(label);self.picture_labels.append(label)
        self.picture_layout.addStretch();self.picture_scroll.setWidget(body);split.addWidget(self.picture_scroll)
        split.setSizes([500,700])
        self.selector.currentIndexChanged.connect(self.select_candidate)
        self.font_size.currentIndexChanged.connect(self.change_font)
        self.change_font();self.selector.setCurrentIndex(1 if self.selector.count()>1 else 0)
        self.select_candidate()
    def change_font(self):
        self.opinion.setStyleSheet('QTextBrowser {font-size:%dpt;padding:12px;}' % self.font_size.currentData())
        font=QFont(self.opinion.font());font.setPointSize(self.font_size.currentData());self.opinion.setFont(font)
        self.opinion.document().setDefaultFont(font)
    def select_candidate(self):
        cid=self.selector.currentData();row=self.rows.get(cid)
        if row is None:
            text=self.result.get('answer_zh') or render_plan(self.result)
        else:
            text=cid+'\n\n'+row.get('observation_zh','')
            text+='\n\n建议核查\n'+'\n'.join(planner.ACTIONS.get(x,x) for x in row.get('requested_checks',[]))
            text+='\n\n补证问题\n'+row.get('question_zh','')
            audit=next((r for r in self.result.get('priority_adjustment',{}).get('rows',[]) if r.get('candidate_id')==cid),None)
            if audit:text+='\n\n当前复核等级\n'+audit.get('display_label','')+'\n'+audit.get('adjustment_reason_zh','')
        self.opinion.setPlainText(text);self.opinion.verticalScrollBar().setValue(0)
        pair=self.images.get(cid)
        for i,label in enumerate(self.picture_labels):
            label.clear()
            if pair is None:label.setText('选择复核框查看图片' if cid is None else '对应局部图片不可用，不使用其他框替代')
            else:label.setPixmap(QPixmap.fromImage(pair[i]).scaled(640,360,Qt.KeepAspectRatio,Qt.SmoothTransformation))

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
    progress=pyqtSignal(str)
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        parent=kwargs.get('parent')
        self.visual_packet=copy.deepcopy(getattr(parent,'_approved_visual_packet',None))
        self.visual_pending=getattr(parent,'_deepseek_pending',None)
        self.thinking_mode=getattr(parent,'thinking_mode',None).currentData() if getattr(parent,'thinking_mode',None) else 'disabled'
        if parent is not None and hasattr(parent,'_on_cloud_progress'):self.progress.connect(parent._on_cloud_progress)
    def run(self):
        try:
            self.progress.emit('正在准备复核输入')
            if self.visual_packet is not None:private.validate(self.visual_packet,self.visual_pending)
            diagnostics={}
            settings=thinking.configured_settings(planner.backend.load_settings(),self.thinking_mode)
            def transport(req,timeout):
                self.progress.emit('请求正在发送，等待大模型判断')
                response=planner.backend._post_json(req,timeout)
                self.progress.emit('服务已响应 · 请求成功，正在校验判断结果')
                return response
            sender=thinking.thinking_sender(self.thinking_mode,transport,
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
        self._reading_result=None;self._reading_images={};self._reading_dialogs=[]
        self.connection_worker=None
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
        self._annotate_optional_tabs()

    def _annotate_optional_tabs(self):
        pages={}
        for name,message in [
            ('port_hint_view','端口局部提示为可选工具。展开左侧“可选工具”，勾选“启用可选端口提示”，完成主检测后点击“生成端口局部提示”。'),
            ('port_comparison_panel','此页显示端口局部原图对比。先生成端口局部提示，再从下方列表选择候选区域查看；未生成时列表为空。'),
            ('rescue_view','独立端口补漏默认关闭。展开左侧“可选工具”，勾选“启用独立端口补漏”，完成主检测后点击“检查端口补漏”。')]:
            view=getattr(self,name,None)
            if view is None:continue
            index=self.tabs.indexOf(view)
            if index<0:continue
            title=self.tabs.tabText(index);self.tabs.removeTab(index)
            page=QWidget();layout=QVBoxLayout(page);layout.setContentsMargins(8,8,8,8)
            note=QLabel(message);note.setWordWrap(True);note.setMaximumHeight(80)
            note.setStyleSheet('padding:10px;background:#edf4f1;color:#425b50;border-radius:6px;')
            layout.addWidget(note);layout.addWidget(view,1);view.show()
            self.tabs.insertTab(index,page,title)
            pages[view]=page
        original_select=self.tabs.setCurrentWidget
        self.tabs.setCurrentWidget=lambda widget:original_select(pages.get(widget,widget))

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
            if control is self.deepseek_answer_label:
                self.deepseek_answer_scroll=QScrollArea()
                self.deepseek_answer_scroll.setWidgetResizable(True)
                self.deepseek_answer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                self.deepseek_answer_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                self.deepseek_answer_scroll.setMinimumHeight(160)
                self.deepseek_answer_scroll.setMaximumHeight(260)
                self.deepseek_answer_scroll.setStyleSheet('QScrollArea {border:1px solid #d9e4de;border-radius:6px;background:#ffffff;}')
                control.setMaximumHeight(16777215)
                control.setWordWrap(True)
                control.setAlignment(Qt.AlignTop|Qt.AlignLeft)
                control.setTextInteractionFlags(Qt.TextSelectableByMouse)
                control.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Minimum)
                control.setContentsMargins(8,8,8,8)
                self.deepseek_answer_scroll.setWidget(control)
                layout.addWidget(self.deepseek_answer_scroll)
                self.reading_button=QPushButton('放大阅读 · 意见与图片对照')
                self.reading_button.setEnabled(False);self.reading_button.clicked.connect(self.open_review_reading)
                layout.addWidget(self.reading_button)
            else:layout.addWidget(control)
        self.deepseek_question_input.hide()
        self.cloud_dialog=QDialog(self);self.cloud_dialog.setWindowTitle('DeepSeek 工作台');self.cloud_dialog.resize(1040,850)
        work_layout=QVBoxLayout(self.cloud_dialog)
        intro=QLabel('本地检测完成后，在这里设置连接、检查外发内容并查看逐框复核结果。');intro.setWordWrap(True);work_layout.addWidget(intro)
        from PyQt5.QtWidgets import QTabWidget
        self.cloud_tabs=QTabWidget();work_layout.addWidget(self.cloud_tabs,1)
        settings_scroll=QScrollArea();settings_scroll.setWidgetResizable(True);settings_scroll.setWidget(self.cloud_body)
        self.cloud_tabs.addTab(settings_scroll,'连接与输入设置')
        result_page=QWidget();result_layout=QVBoxLayout(result_page)
        for control in [self.deepseek_review_button,self.deepseek_answer_scroll,self.reading_button,self.priority_table]:
            layout.removeWidget(control);result_layout.addWidget(control)
        result_scroll=QScrollArea();result_scroll.setWidgetResizable(True);result_scroll.setWidget(result_page)
        self.cloud_tabs.addTab(result_scroll,'复核结果与图片')
        self.cloud_open_button=QPushButton('打开 DeepSeek 工作台');self.cloud_open_button.setEnabled(False)
        self.cloud_open_button.clicked.connect(self.show_cloud_workspace)
        self.result_card.layout().addWidget(self.cloud_switch);self.result_card.layout().addWidget(self.cloud_open_button)
        self.cloud_body.hide();self.cloud_switch.toggled.connect(self.toggle_cloud_workspace)
        feedback=QWidget();row=QHBoxLayout(feedback);row.setContentsMargins(0,0,0,0)
        self.cloud_spinner=gui.BusyIndicator(feedback)
        self.cloud_status=QLabel('尚未发送 · 完成本地检测后可开始复核');self.cloud_status.setWordWrap(True)
        row.addWidget(self.cloud_spinner);row.addWidget(self.cloud_status,1);result_layout.insertWidget(0,feedback)
        self.cloud_elapsed=QTimer(self);self.cloud_elapsed.setInterval(1000)
        self.cloud_elapsed.timeout.connect(self._refresh_cloud_elapsed)
        self._cloud_started=None;self._cloud_phase=''
        self.connection_test_button=QPushButton('测试模型连接（仅文字，不发图片）')
        self.connection_test_button.clicked.connect(self.start_connection_test);layout.insertWidget(1,self.connection_test_button)
        self.connection_status=QLabel('尚未测试');self.connection_status.setWordWrap(True);layout.insertWidget(2,self.connection_status)
        try:
            cfg=planner.backend.load_settings();model_notice=QLabel('当前配置模型：'+cfg.model)
        except Exception:model_notice=QLabel('模型配置不可用，请核对配置文件。')
        layout.insertWidget(3,model_notice)
        layout.setAlignment(Qt.AlignTop)

    def show_cloud_workspace(self):
        if not self.cloud_switch.isChecked():return
        self.cloud_dialog.show();self.cloud_dialog.raise_();self.cloud_dialog.activateWindow()

    def toggle_cloud_workspace(self,enabled):
        self.cloud_body.setVisible(enabled);self.cloud_open_button.setEnabled(enabled)
        if enabled:self.show_cloud_workspace()
        else:self.cloud_dialog.hide()

    def start_connection_test(self):
        if not self.cloud_switch.isChecked() or self.connection_worker is not None:return
        if self.deepseek_worker is not None and self.deepseek_worker.isRunning():return
        try:settings=planner.backend.load_settings()
        except Exception:self.connection_status.setText('模型配置不可用，请检查配置文件。');return
        self.connection_test_button.setEnabled(False);self.connection_status.setText('正在测试模型连接，请稍候…（网络超时设置20秒）')
        worker=ConnectionProbeWorker(settings,self.deepseek_api_input.text().strip() or None,self)
        self.connection_worker=worker
        worker.completed.connect(self.finish_connection_test);worker.finished.connect(self._clear_connection_worker)
        worker.start()

    def finish_connection_test(self,result):
        self.connection_status.setText(result['message'])
        self.connection_status.setStyleSheet('color:#146b4a;' if result['ok'] else 'color:#94601e;')

    def _clear_connection_worker(self):
        if self.connection_worker is not None:self.connection_worker.deleteLater()
        self.connection_worker=None;self.connection_test_button.setEnabled(True)

    def closeEvent(self,event):
        if self.connection_worker is not None and self.connection_worker.isRunning():
            self.connection_status.setText('连接测试正在结束，请稍后再关闭主窗口。');event.ignore();return
        self.cloud_dialog.close();super().closeEvent(event)

    def _capture_reading_result(self,result,pending):
        self._reading_result=None;self._reading_images={};self.reading_button.setEnabled(False)
        if result.get('status')!='ok':return
        try:
            before=private.source_binding(pending)
            ref=private.image(pending['report']['reference']);ins=private.image(Path(pending['output'])/'aligned.jpg')
            if ref.size()!=ins.size():raise ValueError('Frames differ')
            from PyQt5.QtCore import QRect
            import math
            images={}
            for i,candidate in enumerate(pending['candidates'],1):
                x1,y1,x2,y2=planner.backend._candidate_xyxy(candidate)
                if not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or not 0<=x1<x2<=ref.width() or not 0<=y1<y2<=ref.height():continue
                pad=12;left=max(0,math.floor(x1)-pad);top=max(0,math.floor(y1)-pad)
                rect=QRect(left,top,min(ref.width(),math.ceil(x2)+pad)-left,min(ref.height(),math.ceil(y2)+pad)-top)
                pair=[]
                for image in [ref,ins]:
                    crop=image.copy(rect);p=QPainter(crop);p.setPen(QPen(QColor('#e48516'),2))
                    p.drawRect(QRect(round(x1-left),round(y1-top),round(x2-x1),round(y2-y1)));p.end();pair.append(crop)
                images[f'candidate_{i:03d}']=pair
            if private.source_binding(pending)!=before:raise ValueError('Inputs changed')
            self._reading_result=copy.deepcopy(result);self._reading_images=images
            self._reading_output=pending['output'];self.reading_button.setEnabled(True)
        except (OSError,ValueError,KeyError,TypeError):
            # Missing photos must never prevent the accepted text result being read.
            self._reading_result=copy.deepcopy(result);self._reading_output=pending['output'];self.reading_button.setEnabled(True)

    def open_review_reading(self):
        if self._reading_result is None or self.current_output!=self._reading_output:return
        dialog=ReviewReadingDialog(self._reading_result,self._reading_images,self)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        self._reading_dialogs.append(dialog)
        dialog.destroyed.connect(lambda:self._reading_dialogs.remove(dialog) if dialog in self._reading_dialogs else None)
        dialog.show()

    def _on_cloud_progress(self,message):
        if self._cloud_started is None:self._cloud_started=time.monotonic()
        self._cloud_phase=message;self.cloud_spinner.set_running(True)
        self.cloud_elapsed.start();self.cloud_status.setStyleSheet('color:#245f82;')
        self._refresh_cloud_elapsed()
        self._set_stage(message,True)

    def _refresh_cloud_elapsed(self):
        seconds=int(time.monotonic()-self._cloud_started) if self._cloud_started is not None else 0
        self.cloud_status.setText(self._cloud_phase+' · 已等待 '+str(seconds)+' 秒')

    def _finish_cloud_feedback(self,ok=False):
        self.cloud_elapsed.stop();self.cloud_spinner.set_running(False);self._cloud_started=None
        self.cloud_status.setText('发送成功 · 大模型复核完成' if ok else '复核未完成 · 保留本地结果，请查看下方说明')
        self.cloud_status.setStyleSheet('color:#146b4a;' if ok else 'color:#94601e;')

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
        self._reading_result=None;self._reading_images={};self.reading_button.setEnabled(False)
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
        self._reading_result=None;self._reading_images={};self.reading_button.setEnabled(False)
        self._finish_cloud_feedback(False)
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
            self._capture_reading_result(result,pending)
            self.cloud_tabs.setCurrentIndex(1)
            self._finish_cloud_feedback(True)
            level=audit['overall_priority']
            self._set_stage('大模型复核完成 · '+(priority.LABELS[level] if level else '无候选'),False)

    def start_deepseek_mask_review(self):
        if not self.cloud_switch.isChecked():return
        if self.connection_worker is not None:return
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
            self._on_cloud_progress('正在准备发送')
            super().start_deepseek_mask_review()
            if self.deepseek_worker is not None:
                self.input_mode.setEnabled(False);self.preview_button.setEnabled(False)
                self.cloud_switch.setEnabled(False);self.thinking_mode.setEnabled(False);self.save_key_button.setEnabled(False)
            else:self._finish_cloud_feedback(False)

def main():
    gui.DeepSeekMaskReviewWorker=PlannedWorker
    gui.DINOReview=PlannedWindow
    entry.main()

if __name__=='__main__':main()
