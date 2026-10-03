# 智接 Inspection Agent 内核

这个目录是控制柜/线束与可控端子板演示共用的任务层。它包装现有视觉报告，保存工具调用、机器证据、人工结论、修改建议和复检历史。

当前边界：

- DINO、传统视觉和 SAM3 候选只进入 `machine_evidence`，不会自动升级为故障结论。
- 有限拓扑引擎只比较已声明的端子连接，不从像素推断电气连续性。
- 低置信度连接保留为 `insufficient_evidence`。
- 修改建议必须基于人工确认，并绑定可追溯的 `evidence_ids`。
- 示例拓扑是合成契约示例，不是现场数据或准确率证据。

正式 PyQt 入口完成视觉分析后，会在当次输出目录写入
`agent_task.json`。操作员可以在界面中记录复核人、人工结论和修改
建议；若工单处于“等待复检”状态，更换待检图片并再次点击检测按钮，
新报告会追加到原工单的 `reinspection` 历史中。

## 从现有视觉报告创建工单

```powershell
.\.venv\Scripts\python.exe -B tools\inspection_agent_cli.py start `
  --task-id cabinet-demo-001 `
  --scene-type cabinet_harness `
  --reference reference.png `
  --inspection inspection.png `
  --visual-report output\template_reviews\RUN_ID\report.json `
  --output output\agent_tasks\cabinet-demo-001\agent_task.json
```

## 人工复核与修改指导

```powershell
.\.venv\Scripts\python.exe -B tools\inspection_agent_cli.py review `
  --task output\agent_tasks\cabinet-demo-001\agent_task.json `
  --reviewer operator-01 `
  --outcome confirmed_difference `
  --notes "确认存在一根游离线" `
  --candidate green_01

.\.venv\Scripts\python.exe -B tools\inspection_agent_cli.py guidance `
  --task output\agent_tasks\cabinet-demo-001\agent_task.json `
  --instruction "按标准图复核该线端子后重新接入，并重新拍摄" `
  --evidence-id green_01
```

## 有限拓扑比较

```powershell
.\.venv\Scripts\python.exe -B tools\inspection_agent_cli.py topology `
  --task output\agent_tasks\bench-demo-001\agent_task.json `
  --expected config\topology_demo\bench_expected.example.json `
  --observed config\topology_demo\bench_observed_mismatch.example.json
```

拓扑示例仅用于验证 Node/Edge 契约。后续必须用真实端子板照片、人工标注端子与连接，才能评价视觉到拓扑的识别效果。

## SAM3 可见线段到有限拓扑

现有 `extract_sam3_visible_segment_endpoints.py` 已把每个独立 SAM3 掩膜骨架化并
提取两个可见端点。新增适配器只在一条独立线段的两个端点分别且唯一落入两个
不同的人工端子 ROI 时生成候选边：

```powershell
.\.venv\Scripts\python.exe -B tools\sam3_topology_adapter.py `
  --endpoint-report config\topology_demo\sam_endpoint_report.example.json `
  --terminal-map config\topology_demo\sam_terminal_map.example.json `
  --expected config\topology_demo\sam_expected.example.json `
  --output output\sam_topology_bridge_20260926\controlled_contract_report.json
```

端点未命中、同时命中重叠 ROI、两端落入同一 ROI 或端点数不等于 2 时，记录会
保留在审计报告中但不生成边。多个 SAM3 线段支持同一端子对时合并为一条边，
保留全部 `evidence_ids` 并取最高原始 SAM3 分数作为置信度。适配器不会抬高分数，
也不声称物理线缆身份、隐藏路径、电气连续性或自动故障结论。

`wrong2_visible_anchor_map.example.json` 仅用于证明现有 `wrong2` SAM3 端点产物
可以进入同一个图合同；其中节点明确是可见锚点而非电气端子真值，不能作为现场
拓扑准确率证据。

## Mendeley 多正常参考判定

固定单一正常参考会把合法接线配置差异也作为候选。当前隔离评测改为只用
`train01` 的 120 张正常图建立 DINO 空间描述子参考库，以最近 3 个正常参考的
平均余弦距离作为图像级分数；门槛只在 `val01` 选择，随后固定应用于
`test01`：

```powershell
.\.venv\Scripts\python.exe -B tools\evaluate_mendeley_normal_bank.py `
  --output output\mendeley_normal_bank_20260926_v1 `
  --k 3 --minimum-sensitivity 0.90

.\.venv\Scripts\python.exe -B tools\classify_mendeley_normal_bank.py `
  --model output\mendeley_normal_bank_20260926_v1\normal_bank_model.npz `
  --image "data\external_datasets\mendeley_electrical_wiring_faults\Predictive Maintenance for Electrical Wiring Faults\images\test01\normal_006.JPG"
```

该门只输出 `normal_like_reference_bank` 或
`possible_fault_manual_review`。原 DINO/传统差异候选仍负责定位人工复核区域；
它不会输出故障类型、端点或拓扑。数据来自同一机箱和高度相关的受控拍摄，因而
数据集内结果不能外推为现场准确率；此外 test01 在更早实验中已被查看，本次结果
是从此冻结规则后的回顾性评估，不是全新未见测试集。

隔离的 Mendeley 两级复核入口先运行上述图像级门；只有疑似故障图才运行定位，
默认使用 `train01/normal_073.JPG` 作固定参照、最多保留 6 个候选。正常类判定
返回零候选，但不等于证明无故障。这个入口不修改控制柜 GUI 的检测逻辑：

```powershell
.\.venv\Scripts\python.exe -B tools\run_mendeley_review_cascade.py `
  --image "data\external_datasets\mendeley_electrical_wiring_faults\Predictive Maintenance for Electrical Wiring Faults\images\test01\damaged_005.JPG" `
  --model output\mendeley_normal_bank_20260926_v1\normal_bank_model.npz `
  --train-normal-dir "data\external_datasets\mendeley_electrical_wiring_faults\Predictive Maintenance for Electrical Wiring Faults\images\train01" `
  --output output\mendeley_reference_localization_20260927\cascade_damaged_005.json
```

`tools/audit_mendeley_grouped_bank.py` 可用数字编号邻近排除法做相关性压力测试；
编号距离不是实际拍摄批次标签。`tools/evaluate_mendeley_reference_localization.py`
把原始标注框经同一类单应变换映射到参照坐标，再同时报告图像命中、标注框
重叠、IoU 和候选负担。参数是在 `val01` 上选择的，`test01` 只作同机箱回顾性
复核，不能据此声称现场准确率提升。

2026-09-27 复核：`val01` 15 张故障图，候选上限由 3 提高到 6 时，碰到的
标注框由 108/245 增至 146/245；该上限 6 配置在 `test01` 为 163/242，低于旧
固定参照方案的 165/242，而且命中任一标注框的候选仅 37/69。两级门在同机箱
`test01` 图像分类仍为 30/30，但定位精度未验证有净提升。详见
`output/mendeley_reference_localization_20260927/` 下的逐图 JSON 报告。

随后在 `val01` 做候选后处理离线审计：既有报告中原排序前 5 个与前 6 个的
源框命中均为 146/245，候选由 70 减至 65；`test01` 回顾性报告的前 5 个
也是 163/242，候选由 69 减至 65。但实际把底层预算改为 5 并重跑单图时，
输出并非预算 6 结果的前 5 个，说明该离线分析不能直接代表运行效果。因此
完整重跑 `val01` 的真实底层预算 5 后，源框命中反而降至 129/245，
IoU>=0.1 由 23/245 降至 17/245，虽候选从 70 降至 65，仍不划算。
所以保留预算 6，不接入裁剪；缩框、按面积或分数重排也会漏掉更多源框。
审计工具为 `tools/audit_mendeley_candidate_postprocess.py`。

## Mendeley 定位误差与局部热图细化审计（2026-09-27）

`tools/audit_mendeley_localization_errors.py` 对 `val01` 的预算 6 报告逐图统计：
70 个候选中有 30 个不与来源标注框重叠，候选之间达到较高相互覆盖的仅 2 对；
部分跨 whole/tile/refinement 合并框面积是所碰小标注框的数百倍。因此当前主要
问题不是大量重复框，而是不同尺度证据的并框太宽、同时图中合法结构也有强差异。
来源框未必穷尽全部真实异常，故「不重叠」只用于审计，不能直接称假阳性。

隔离实验用 `tools/probe_mendeley_local_heat.py` 快取原有 DINO/传统融合热图，
再由 `tools/evaluate_mendeley_local_refinement.py` 在每个候选内选择连通高分区。
快取输出与原报告的 70 个候选及指标完全一致。仅保留细框时，最温和的
验证集设置把源框命中由 146/245 降为 142/245、IoU>=0.5 由 4 降为 3；
若同时保留原框，源框命中仅升到 148/245，但候选数由 70 翻倍到 140。
故本实验**未接入正式入口**，也未据此调 `test01`；正式入口仍使用原预算 6。
逐图审计和完整参数对比在 `output/mendeley_local_refinement_20260927/`。

比较报告现在同时给出 `topology_distance`：它计算 Expected 与高置信度
Observed 边集合的归一化对称差，`0` 表示已声明连接完全一致，`1` 表示没有
任何边重合。这个轻量指标受 Image2Net 的 Netlist Edit Distance 启发，但不是
其异构图 GED/NED 的复刻；低置信度边不会被硬算作正确或错误，而会单独留在
`uncertain_connections` 并令任务进入 `insufficient_evidence`。

## 外部资源核验

```powershell
.\.venv\Scripts\python.exe -B tools\audit_external_resources.py `
  --manifest external_resources\manifest.json `
  --output external_resources\audit_report.json
```

该命令只核对资源完整性、许可元数据和样本配对，不把电路图、运动线缆样本
或合成数据当作真实端子板识别准确率。

## 当前外部数据基线

Image2Net 的轻量适配把每个有序器件端口映射为 `component_port -> net`
连接，再用现有拓扑比较器检查原样、删一条连接和改接一条端口：

```powershell
.\.venv\Scripts\python.exe -B tools\evaluate_image2net_topology.py `
  --golden-dir external_resources\image2net\validation\golden `
  --output output\external_resource_baselines\image2net_topology_report.json
```

该适配依赖器件顺序稳定，只是连接图合同基线，不等价于 Image2Net 的异构图
同构与 GED/NED，也不包含图像识别。

MovingCables 基线仅用 clip `0003`：前半拟合平衡 HSV 颜色直方图，后半选择
阈值和形态学参数；clip `0006` 只做一次独立评估：

```powershell
.\.venv\Scripts\python.exe -B tools\evaluate_movingcables_baseline.py `
  --archive external_resources\movingcables_sample\MovingCables_sample.tar `
  --calibration-clip 0003 --evaluation-clip 0006 `
  --output-dir output\external_resource_baselines\movingcables
```

它输出像素 Precision/Recall/IoU/F1、逐帧 IoU 和三张人工复核叠加图。
这些指标仅描述 MovingCables 样例域的“全部线缆语义掩膜”，不包含单根线
实例分离、端点、端子归属、连接拓扑或柜体准确率。
