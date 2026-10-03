"""Generic no-marker assembly review: stable template localization + check ROIs."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea, QTabWidget, QTextEdit, QVBoxLayout, QWidget

from dimm_review_app_fixed import RoiCanvas
from dimm_upper_right_review import read_image, write_image
from assembly_anchor_review_app import ZoomImageView

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "template_review_recipe.json"
OUT = ROOT / "output" / "template_reviews"


class RectPicker(QDialog):
    def __init__(self, image: np.ndarray, title: str, hint: str, parent: QWidget) -> None:
        super().__init__(parent); self.setWindowTitle(title); self.setModal(True)
        screen = QApplication.primaryScreen().availableGeometry(); self.canvas = RoiCanvas(image, max(700, screen.width()-160), max(500, screen.height()-230))
        info = QLabel(hint); info.setWordWrap(True)
        scroll = QScrollArea(); scroll.setWidget(self.canvas); scroll.setWidgetResizable(False)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); buttons.button(QDialogButtonBox.Ok).setText("确认"); buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.confirm); buttons.rejected.connect(self.reject)
        box = QVBoxLayout(self); box.addWidget(info); box.addWidget(scroll, 1); box.addWidget(buttons)
        self.resize(min(self.canvas.display_width+70, screen.width()-160), min(self.canvas.display_height+150, screen.height()-230))
    def confirm(self) -> None:
        if self.canvas.normalized_roi() is None: QMessageBox.warning(self, "尚未框选", "请拖动鼠标框选一个区域。"); return
        self.accept()
    def roi(self) -> list[float] | None: return self.canvas.normalized_roi()


def pixels(image: np.ndarray, roi: list[float]) -> tuple[int, int, int, int]:
    h,w=image.shape[:2]; return round(roi[0]*w),round(roi[1]*h),round(roi[2]*w),round(roi[3]*h)


def normalize(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.normalize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),None,0,255,cv2.NORM_MINMAX),(5,5),0)


def locate(reference: np.ndarray, inspection: np.ndarray, template_roi: list[float]) -> tuple[np.ndarray | None, dict[str, Any]]:
    x1,y1,x2,y2=pixels(reference,template_roi); template=reference[y1:y2,x1:x2]
    orb=cv2.ORB_create(nfeatures=4000, fastThreshold=7)
    kp_template,desc_template=orb.detectAndCompute(cv2.cvtColor(template,cv2.COLOR_BGR2GRAY),None)
    kp_image,desc_image=orb.detectAndCompute(cv2.cvtColor(inspection,cv2.COLOR_BGR2GRAY),None)
    report={"method":"template_orb_ransac","template_keypoints":len(kp_template),"inspection_keypoints":len(kp_image),"matches":0,"inliers":0}
    if desc_template is None or desc_image is None: report["reason"]="template_has_too_little_texture"; return None,report
    pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_image,desc_template,k=2)
    good=[a for a,b in pairs if a.distance < .70*b.distance]; report["matches"]=len(good)
    if len(good)<12: report["reason"]="too_few_template_matches"; return None,report
    source=np.float32([kp_image[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    destination=np.float32([[kp_template[m.trainIdx].pt[0]+x1,kp_template[m.trainIdx].pt[1]+y1] for m in good]).reshape(-1,1,2)
    H,mask=cv2.findHomography(source,destination,cv2.RANSAC,4.0)
    if H is None or mask is None: report["reason"]="no_template_transform"; return None,report
    report["inliers"]=int(mask.sum()); report["inlier_ratio"]=round(report["inliers"]/len(good),4)
    projected=cv2.perspectiveTransform(source,H); error=np.linalg.norm(projected.reshape(-1,2)-destination.reshape(-1,2),axis=1)
    report["median_reprojection_error"]=round(float(np.median(error[mask.ravel().astype(bool)])),2)
    if report["inliers"]<12 or report["inlier_ratio"]<.35 or report["median_reprojection_error"]>3.0:
        report["reason"]="unstable_template_location_retake_photo"; return None,report
    h,w=reference.shape[:2]; return cv2.warpPerspective(inspection,H,(w,h)),report


def inspect_regions(reference: np.ndarray, aligned: np.ndarray, rois: list[list[float]]) -> tuple[np.ndarray,list[dict[str,Any]]]:
    overlay=aligned.copy(); heat=np.zeros_like(reference); regions=[]
    for roi_index,roi in enumerate(rois,1):
        x1,y1,x2,y2=pixels(reference,roi); diff=cv2.absdiff(normalize(reference[y1:y2,x1:x2]),normalize(aligned[y1:y2,x1:x2])); threshold=max(25.,float(np.percentile(diff,99.2)))
        mask=np.uint8(diff>=threshold)*255; kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)); mask=cv2.dilate(cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel),kernel,iterations=2)
        colored=cv2.applyColorMap(diff,cv2.COLORMAP_JET); heat[y1:y2,x1:x2]=colored
        cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,215,255),2)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        minimum=max(120,int(mask.shape[0]*mask.shape[1]*.00045))
        for contour in contours:
            area=float(cv2.contourArea(contour))
            if area<minimum: continue
            x,y,w,h=cv2.boundingRect(contour); item={"check_region":roi_index,"left":x+x1,"top":y+y1,"right":x+x1+w,"bottom":y+y1+h,"area":round(area,1),"difference_score":round(float(diff[y:y+h,x:x+w].mean()),2)}; regions.append(item)
    regions.sort(key=lambda r:r["area"]*r["difference_score"],reverse=True)
    for number,r in enumerate(regions,1):
        left,top,right,bottom=(int(r[k]) for k in ("left","top","right","bottom")); cv2.rectangle(overlay,(left,top),(right,bottom),(0,0,255),4); cv2.putText(overlay,str(number),(left,max(30,top-8)),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,0,255),3)
    return overlay,heat,regions


class TemplateReview(QMainWindow):
    def __init__(self) -> None:
        super().__init__(); self.recipe=self.load_recipe(); self.setWindowTitle("AI 装配视觉复核｜无标记模板定位"); self.resize(1450,900); self.build(); self.reference.setText(self.recipe.get("reference_image","")); self.refresh()
    def load_recipe(self) -> dict[str,Any]:
        return json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {"reference_image":"","template_roi":None,"check_rois":[]}
    def save_recipe(self) -> None:
        CONFIG.parent.mkdir(parents=True,exist_ok=True); CONFIG.write_text(json.dumps(self.recipe,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    def build(self) -> None:
        root=QWidget(); self.setCentralWidget(root); layout=QHBoxLayout(root); left=QWidget(); left.setMinimumWidth(390); left.setMaximumWidth(500); controls=QVBoxLayout(left)
        text=QLabel("首次设置：框选一个始终可见、纹理清晰的稳定模板，再添加要检查的插头/DIMM/线缆区域。日常检测只自动定位模板，并只在检查区画红框。"); text.setWordWrap(True); text.setStyleSheet("background:#eef6ff;padding:12px;border-radius:6px"); controls.addWidget(text)
        group=QGroupBox("正确参考图与检查配方"); form=QFormLayout(group); self.reference=QLineEdit(); self.inspection=QLineEdit()
        for title,edit,fn in (("正确参考图",self.reference,self.choose_reference),("待检图片",self.inspection,self.choose_inspection)):
            row=QWidget(); box=QHBoxLayout(row); box.setContentsMargins(0,0,0,0); button=QPushButton("选择"); button.clicked.connect(fn); box.addWidget(edit); box.addWidget(button); form.addRow(title,row)
        controls.addWidget(group)
        for title,fn in (("框选稳定定位模板",self.set_template),("添加检查区域",self.add_region),("撤回最后检查区",self.undo_region),("重置模板和检查区",self.reset_recipe)):
            button=QPushButton(title); button.clicked.connect(fn); controls.addWidget(button)
        run=QPushButton("自动定位并生成红框复核结果"); run.setMinimumHeight(48); run.setStyleSheet("font-size:16px;font-weight:bold;background:#1677ff;color:white"); run.clicked.connect(self.run); controls.addWidget(run)
        self.status=QLabel(); self.status.setWordWrap(True); controls.addWidget(self.status); controls.addStretch(); layout.addWidget(left)
        right=QWidget(); box=QVBoxLayout(right); self.tabs=QTabWidget(); self.views=[ZoomImageView("尚未检测") for _ in range(3)]
        for view,title in zip(self.views,("模板对齐图","检查区红框","检查区热图")): self.tabs.addTab(view,title)
        box.addWidget(self.tabs,3); self.report=QTextEdit(); self.report.setReadOnly(True); box.addWidget(self.report,1); layout.addWidget(right,1)
    def refresh(self) -> None:
        template="已设置" if self.recipe.get("template_roi") else "未设置"; self.status.setText(f"模板：{template}；检查区：{len(self.recipe.get('check_rois',[]))} 个。")
    def choose_reference(self) -> None:
        path,_=QFileDialog.getOpenFileName(self,"选择正确参考图",str(Path.home()/"Desktop"),"图片 (*.jpg *.jpeg *.png *.bmp)")
        if path: self.reference.setText(path); self.recipe={"reference_image":path,"template_roi":None,"check_rois":[]}; self.save_recipe(); self.refresh()
    def choose_inspection(self) -> None:
        path,_=QFileDialog.getOpenFileName(self,"选择待检图",str(Path.home()/"Desktop"),"图片 (*.jpg *.jpeg *.png *.bmp)")
        if path:self.inspection.setText(path)
    def reference_image(self) -> np.ndarray | None:
        path=Path(self.reference.text().strip());
        if not path.is_file(): QMessageBox.warning(self,"缺少参考图","请先选择正确参考图。"); return None
        self.recipe["reference_image"]=str(path); return read_image(path)
    def set_template(self) -> None:
        image=self.reference_image();
        if image is None:return
        picker=RectPicker(image,"框选稳定定位模板","选择外壳印字、固定纹理、未被遮挡的插孔边角等稳定区域；不要选择线缆、插头或待检部位。",self)
        if picker.exec_()==QDialog.Accepted:self.recipe["template_roi"]=picker.roi(); self.save_recipe(); self.refresh()
    def add_region(self) -> None:
        image=self.reference_image();
        if image is None:return
        picker=RectPicker(image,"添加检查区域","框选一个真正需要检查的区域，例如一个插头、一个 DIMM 槽组或一个端子。",self)
        if picker.exec_()==QDialog.Accepted:self.recipe.setdefault("check_rois",[]).append(picker.roi()); self.save_recipe(); self.refresh()
    def undo_region(self) -> None:
        if self.recipe.get("check_rois"):self.recipe["check_rois"].pop();self.save_recipe();self.refresh()
    def reset_recipe(self) -> None:
        if QMessageBox.question(self,"确认重置","删除当前模板和所有检查区域？",QMessageBox.Yes|QMessageBox.No)==QMessageBox.Yes:self.recipe["template_roi"]=None;self.recipe["check_rois"]=[];self.save_recipe();self.refresh()
    def run(self) -> None:
        ref_path=Path(self.reference.text().strip()); test_path=Path(self.inspection.text().strip())
        if not ref_path.is_file() or not test_path.is_file() or not self.recipe.get("template_roi") or not self.recipe.get("check_rois"): QMessageBox.warning(self,"未完成设置","请选择两张图片，并设置一个模板和至少一个检查区。");return
        reference,inspection=read_image(ref_path),read_image(test_path); aligned,alignment=locate(reference,inspection,self.recipe["template_roi"])
        output=OUT/datetime.now().strftime("%Y%m%d_%H%M%S");output.mkdir(parents=True,exist_ok=True)
        if aligned is None:
            report={"decision":"retake_photo","reference":str(ref_path),"inspection":str(test_path),"alignment":alignment,"reason":"稳定模板无法可靠定位，请按固定拍摄姿态重拍，或换一个纹理更清晰且无遮挡的模板。"};(output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");self.report.setPlainText(json.dumps(report,ensure_ascii=False,indent=2));self.status.setText("模板定位失败：请重拍或更换模板。");return
        overlay,heat,regions=inspect_regions(reference,aligned,self.recipe["check_rois"]);write_image(output/"aligned.jpg",aligned);write_image(output/"anomaly_boxes.jpg",overlay);write_image(output/"check_heatmap.jpg",heat)
        report={"decision":"manual_review_required","reference":str(ref_path),"inspection":str(test_path),"alignment":alignment,"template_roi":self.recipe["template_roi"],"check_region_count":len(self.recipe["check_rois"]),"review_regions":regions,"note":"只在用户定义的检查区内产生红框；红框仍须人工确认。"};(output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        for view,name in zip(self.views,("aligned.jpg","anomaly_boxes.jpg","check_heatmap.jpg")):view.load(output/name)
        self.tabs.setCurrentIndex(1);self.report.setPlainText(json.dumps(report,ensure_ascii=False,indent=2));self.status.setText(f"模板定位成功；{len(self.recipe['check_rois'])} 个检查区中发现 {len(regions)} 个红框。结果：{output}")

def main() -> None:
    app=QApplication(sys.argv);app.setStyle("Fusion");window=TemplateReview();window.show();raise SystemExit(app.exec_())
if __name__=="__main__":main()
