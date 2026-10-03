# Mendeley `class 1` 线缆损坏网格 POC

目标是对同一 Dell 机箱、近似同一拍摄视角的单张全景图，标出可能含有可见线缆损坏的**局部网格候选**，供人工复核。

原始 `class 1` 经图像审计解释为损坏线缆的小区域。由于损坏点分布广，使用不重叠的 6×4 相机固定网格；网格位置与所有标注无关。每个网格仅判断是否包含至少一个 `class 1` 框，因此不能给出精确损坏点位置或数量。

留出测试结果：网格级 `26 TP / 10 FP / 5 FN / 679 TN`，精度 72.2%、召回 83.9%；图像级 5/5 损坏图报警，25 张非损坏图有 2 张误报。报警网格覆盖原始 104 个损坏框中的 96 个（92.3%），但该覆盖率是粗网格语义，不能当作精确框检测率。

它不验证插口状态、走线 A→B 或电气连通。对于误报的断开线缆图，模型会把悬空插头附近复杂线束误判为损坏，因此只能用作人工复核筛查。

```powershell
& E:\PythonProject10\.venv\Scripts\python.exe -B E:\PythonProject10\prototype\mendeley_damage_roi_poc.py `
  --image '...\test01\damaged_032.JPG' `
  --recipe E:\PythonProject10\data\derived\mendeley_damage_grid_roi_20260825\recipe.json `
  --weights E:\PythonProject10\output\mendeley_damage_grid_roi_cpu_20260825\best.pt `
  --output E:\PythonProject10\output\mendeley_damage_grid_roi_inference_example
```
