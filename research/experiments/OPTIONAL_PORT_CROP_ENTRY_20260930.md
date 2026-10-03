# 可选单图端口提示入口（2026-09-30）

`tools/run_optional_port_crop_review.py` 提供默认关闭的单图入口；`inspection_agent/optional_port_crop_review.py` 提供函数。默认 GUI 尚未调用此入口。

启用后沿用固定验收：1280 裁块、步长960、切边16像素弃权、类内 IoU .5 去重、模型输入960、阈值 .25、有效配准覆盖≥.98、提示完整包含在原父框且面积≤父框一半、已有提示 IoU≥.5 去重，每图最多新增1个提示。保留原父框与已有提示。

## 输入和使用

需要源图、参考图、已完成视觉复核产生的上下文 JSON。上下文只含：

- `parents`：参考图坐标的原复核父框。
- `existing_hints`：原有整图端口提示。
- `source_sha256`、`reference_sha256`：绑定当前输入。
- `source_to_reference_homography`：源图到参考图的3×3变换。
- `alignment_reliable`：上游配准质量检查结果。

不会消费标注、真值类别或图名作为提示选择条件。上下文的几何与质量由上游提供，入口不重新估计配准。`--scene` 是调用者声明，入口没有自动场景识别器。

目前限定 `mendeley_pc_wiring_same_camera`，源图/参考图2736×3648，参考图必须与既有 train01/normal_073.JPG 的 SHA 一致；这是当前模型验收范围的准入条件，不是通用化能力。其他机柜或未知场景回退原结果。

调用形式（使用项目虚拟环境；输出目录应可写）：

```powershell
.venv\Scripts\python.exe tools\run_optional_port_crop_review.py --project E:\PythonProject10 --image <源图> --reference <参考图> --context <上下文JSON> --calibration config\port_crop_calibration_frozen_20260930.json --output <输出JSON> --enable --scene mendeley_pc_wiring_same_camera
```

省略 `--enable` 则返回 `status=disabled`，不加载模型。权重、校准文件或提示辅助代码指纹不符，配准不可靠、输入身份不符、几何不符或推理异常则 `status=fallback`、`tile_hints=[]`，返回原父框/提示及原因。`status=applied` 仅表示可选逻辑执行完成，可以仍然没有新增提示。模型权重默认使用固定6轮训练的 `output/port_crop_training_fixed_20260929/full/runs/rectports/weights/best.pt`。

## 验证与定位

Codex 工作区 `artifacts/optional_port_crop_entry_20260930` 保存本轮结果；`experiments/verify_optional_port_crop_review.py` 保存回归入口。固定30图的变换、提示及审计应与既有验收完全一致，6个新增提示、正常新增0。10项故障注入覆盖关闭、场景/配准/源图/参考图不符、奇异H、校准/权重被改、权重缺失和推理失败。

这轮提供接入能力，精度结果仍沿用上一轮5/245→10/245、正常复核候选37→37、故障候选71→77。仅为同设备现有标签的定位匹配；没有新训练、新场景泛化或电气拓扑验收。`prediction_provider` 参数仅用于隔离重放验证，CLI 不开放。

下一步：GUI 如要调用，应显式保留关闭开关，并由原配准流程生成上述上下文。机柜场景先建立端子身份与预期接线表，当前机箱端口模型不直接用于机柜拓扑。
