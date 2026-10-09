"""Opt-in experimental mainline window; overrides worker only in this process."""
import sys
from pathlib import Path

sys.path[:0]=['E:/PythonProject10/models/dinov2','E:/PythonProject10/prototype']
# Preserve the mainline Windows torch-before-Qt import order.
import assembly_auto_review_dino_v2 as entry
import assembly_auto_review_dino as gui
import llm_recheck_planner as planner
import llm_review_priority as priority
from PyQt5.QtWidgets import QMessageBox, QTableWidget, QTableWidgetItem, QAbstractItemView
from PyQt5.QtGui import QColor

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
    def run(self):
        try:
            result=priority.run(self.reference_mask_path,self.inspection_mask_path,self.candidates,api_key=self.api_key)
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
        self.priority_table=QTableWidget(0,5,self)
        self.priority_table.setHorizontalHeaderLabels(['候选','原等级','模型建议','当前等级','调整依据'])
        self.priority_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.priority_table.setMinimumHeight(150)
        self.priority_table.horizontalHeader().setStretchLastSection(True)
        self.result_card.layout().addWidget(self.priority_table)

    def run(self):
        self.priority_table.setRowCount(0)
        super().run()

    def _finish_deepseek_mask_review(self,result):
        pending=self._deepseek_pending
        if pending is None or self.current_output!=pending['output']:return
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
        if self._deepseek_pending is None:return
        answer=QMessageBox.question(self,'确认本次掩膜外发',
            '将本次两张二值布线掩膜、候选坐标和最少差异信息发送至配置的DeepSeek服务。'
            '不发送原照片，但掩膜仍可能暴露布线布局。判断只影响复核优先级，不删除框、不修改连接结论。是否继续？',
            QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if answer==QMessageBox.Yes: super().start_deepseek_mask_review()

def main():
    gui.DeepSeekMaskReviewWorker=PlannedWorker
    gui.DINOReview=PlannedWindow
    entry.main()

if __name__=='__main__':main()
