# Mendeley `class 3` 悬空插头端网格 POC

目标是对同一 Dell 机箱、近似同一拍摄视角的单张全景图，把可能含有悬空/断开插头端的**粗网格区域**标为人工复核候选。

这不是精确插头框检测、已插好确认、线缆 A→B 配对或电气连通验证。

## 数据与评估

- 原始 `class 3` 经图像审计解释为断开的插头端。
- 插头端在训练图中分布到许多位置，不能复用空插口的六个固定接口 ROI；因此使用不重叠的 4×3 相机固定网格。
- 网格仅由画面坐标定义，训练/验证/测试标注均不参与网格位置选择。
- 每个网格只判断是否包含至少一个 `class 3` 端点；同格多个端点合并为一个候选。
- 留出测试：`19 TP / 9 FP / 6 FN / 326 TN`，网格级精度 67.9%、召回 76.0%；图像级 5/5 断开图报警，25 张非断开图中有 3 张误报。

所以它可用于“把图送给人工复核”的粗筛，不能自动宣布某个插头断开。

```powershell
& E:\PythonProject10\.venv\Scripts\python.exe -B E:\PythonProject10\prototype\mendeley_plug_roi_poc.py `
  --image '...\test01\disconnected_016.JPG' `
  --recipe E:\PythonProject10\data\derived\mendeley_plug_grid_roi_20260825\recipe.json `
  --weights E:\PythonProject10\output\mendeley_plug_grid_roi_cpu_20260825\best.pt `
  --output E:\PythonProject10\output\mendeley_plug_grid_roi_inference_example
```
