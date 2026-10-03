# Mendeley 固定插口局部识别 POC

目标是固定 Dell 机箱、近似同一拍摄视角下，标出六个固定区域中**可见的空插口候选**。它不是线缆走线 A→B 识别，也不能验证插头完全插到位或电气连通。

数据策略：只从 `train01` 的原始 `class 4` 框学习六个局部区域；`val01` 仅用于选阈值，`test01` 不参与区域定位、训练或阈值选择。类别 `0` 的语义是“该区域未出现源数据标注的空插口”，不是“已插好”。

已完成的可复现实验：

- 全图 YOLO POC 在留出测试上只找回 6/29 个断开插头、0/29 个空插口，因此弃用。
- 局部 POC 对训练期已覆盖的六个区域，在留出测试的 180 个区域裁剪上得到 `24 TP / 0 FP / 0 FN / 156 TN`。30 张图层面为 `5 TP / 0 FP / 0 FN / 25 TN`。
- 这是一批高度相似拍摄素材的 source-split 结果，不能当作新机箱、不同相机或现场准确率；仍需在陌生拍摄条件上取图验证。

复现：

```powershell
& E:\PythonProject10\.venv\Scripts\python.exe -B E:\PythonProject10\tools\prepare_mendeley_jack_roi_dataset.py `
  --dataset-root 'E:\PythonProject10\data\external_datasets\mendeley_electrical_wiring_faults\Predictive Maintenance for Electrical Wiring Faults' `
  --output E:\PythonProject10\data\derived\mendeley_jack_roi_20260825

& E:\PythonProject10\.venv\Scripts\python.exe -B E:\PythonProject10\tools\train_mendeley_jack_roi_classifier.py `
  --dataset-dir E:\PythonProject10\data\derived\mendeley_jack_roi_20260825 `
  --output E:\PythonProject10\output\mendeley_jack_roi_cpu_20260825
```

对一张匹配的机箱图推理：

```powershell
& E:\PythonProject10\.venv\Scripts\python.exe -B E:\PythonProject10\prototype\mendeley_jack_roi_poc.py `
  --image '...\test01\disconnected_016.JPG' `
  --recipe E:\PythonProject10\data\derived\mendeley_jack_roi_20260825\recipe.json `
  --weights E:\PythonProject10\output\mendeley_jack_roi_cpu_20260825\best.pt `
  --output E:\PythonProject10\output\mendeley_jack_roi_inference_example
```

下一阶段如果要做 A→B：先人工命名六个区域的真实接口与相应线端，再把“允许的端点配对”写为拓扑表；仅凭目前的单张全机箱图和框标注，不可验证线路全程或电气状态。
