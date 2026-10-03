"""Light workbench styling only: no inspection or task-state mutations."""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

STYLE = """
QWidget {font-family: 'Microsoft YaHei', 'Segoe UI'; font-size: 12px; color: #303740;}
QMainWindow, QDialog {background: #f7f8fa;}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
 background: white; border: 1px solid #d6e1dd; border-radius: 5px;
 padding: 5px; selection-background-color: #d3ebe3; selection-color: #164e3f;
}
QLineEdit:focus, QComboBox:focus {border-color: #178264;}
QPushButton {background: #ffffff; border: 1px solid #cfddd7; border-radius: 5px;
 padding: 7px 10px; color: #254c40;}
QPushButton:hover {background: #e9f4ee; border-color: #8ebba7;}
QPushButton:pressed {background: #d8eee3;}
QPushButton:disabled {color: #8c9994; background: #eff2f0; border-color: #e0e6e2;}
QPushButton#workbenchPrimary {background: #146b58; color: white; border-color: #146b58; font-weight: bold;}
QPushButton#workbenchPrimary:hover {background: #105745;}
QPushButton#workbenchPrimary:disabled {background: #b7cbc3; border-color: #b7cbc3;}
QLabel#workbenchBrand {font-size: 15px; font-weight: bold; color: #303740; padding: 2px 0;}
QLabel#workbenchSubtitle {color: #6b7f75; font-size: 11px; padding-bottom: 7px;}
QTabWidget::pane {border: 1px solid #d9e4de; background: white; border-radius: 5px;}
QTabBar::tab {background: #edf3ef; border: 1px solid #d9e4de; padding: 8px 10px; margin-right: 2px; color: #647a6f;}
QTabBar::tab:selected {background: white; color: #146b58; border-bottom: 2px solid #146b58;}
QGroupBox {background: white; border: 1px solid #d9e4de; border-radius: 6px; margin-top: 14px; padding: 10px;}
QGroupBox::title {subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #466a58;}
QCheckBox {spacing: 7px;}
QScrollArea {border: 0; background: #f7f8fa;}
QProgressBar {border: 1px solid #d9e4de; border-radius: 4px; background: #edf3ef; text-align: center;}
QProgressBar::chunk {background: #178264;}
"""

def apply_workbench_theme(window):
    """Keep the existing layout, signals, safety gates and status meanings."""
    window.setWindowTitle('智接 · 线缆复核')
    window.setStyleSheet(STYLE)
    window.run_button.setObjectName('workbenchPrimary')
    window.run_button.setStyleSheet('')
    layout = window.status.parentWidget().layout()
    brand = QLabel('线缆复核', window)
    brand.setObjectName('workbenchBrand')
    layout.insertWidget(0, brand)
    window.tabs.setUsesScrollButtons(True)
    window.tabs.setDocumentMode(True)
    for index in range(window.tabs.count()):
        view = window.tabs.widget(index)
        if hasattr(view, 'setBackgroundBrush'):
            view.setBackgroundBrush(QColor('#fafbfc'))
            if not view.scene.items():
                hint = view.scene.addText('尚未检测', QFont('Microsoft YaHei', 11))
                hint.setDefaultTextColor(QColor('#89919a'))
    window.report.setPlaceholderText('检测记录')
    # Optional tools stay available; folding changes visibility, not enablement.
    result_layout = window.result_card.layout()
    optional_body = QWidget(window.result_card)
    optional_layout = QVBoxLayout(optional_body)
    optional_layout.setContentsMargins(0, 8, 0, 8)
    for name in ('port_hint_switch', 'port_scene_combo', 'port_hint_button',
                 'port_hint_status', 'deepseek_api_input', 'deepseek_question_input',
                 'deepseek_review_button', 'deepseek_answer_label'):
        control = getattr(window, name)
        result_layout.removeWidget(control)
        optional_layout.addWidget(control)
    optional_toggle = QPushButton('+ 可选工具', window.result_card)
    optional_toggle.setStyleSheet('text-align:left; background:transparent; border:0; color:#626b75;')
    def toggle_optional():
        expanded = optional_body.isHidden()
        optional_body.setVisible(expanded)
        optional_toggle.setText(('-' if expanded else '+') + ' 可选工具')
    optional_toggle.clicked.connect(toggle_optional)
    optional_body.hide()
    result_layout.insertWidget(4, optional_toggle)
    result_layout.insertWidget(5, optional_body)
    window.workbench_optional_toggle = optional_toggle
    window.workbench_optional_body = optional_body
    root_layout = window.centralWidget().layout()
    left = window.status.parentWidget()
    root_layout.removeWidget(left)
    scroll = QScrollArea(window)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setMinimumWidth(420)
    scroll.setMaximumWidth(520)
    scroll.setWidget(left)
    root_layout.insertWidget(0, scroll)
    window.workbench_controls_scroll = scroll
