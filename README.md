# 智接 / WireMind — 电气装配复核 Agent

当前项目和全部主要模型（含 3.45 GB SAM3）的备份见
[恢复说明](backup/README.md) 与
[模型 Release](https://github.com/seeleayaka/wire/releases/tag/backup-20261004)。
本仓库是公开的脱敏快照，不含密钥或私人样本。

## 可控思考的大模型复核入口

运行`启动大模型复核版.bat`，启用DeepSeek后可设置密钥、纯掩膜或脱敏局部照片输入，以及关闭、轻量、标准、深度四档思考。轻量为推荐默认；接口只在用户点击并授权后调用。本机记住密钥，不写入报告或仓库。原始候选与正式视觉入口保持不变。

更新后的[比赛材料公开版](competition/submission_materials_20261009/README.md)包含作品亮点、部署说明、测试记录和AI使用声明。公开版不携带私人机柜照片，完整SAM及其余模型仍从原Release恢复。

新增原生候选增强已完成实际验收并接入，沿用原增强入口、默认关闭。
未通过验收的下一轮研究试验仍与主线分开，不会自动启用。
10 月 4 日后续实验、失败结果及仍运行任务的边界见
[研究备份状态](research/evidence/research_backup_status_20261004/STATUS.md)。
SAM3 分片及其余模型 Release 保持原样，新增研究代码不代表正式升级。

以下为保留的项目原说明，部分早期命令需要自行准备样本和本机配置。

First-stage pipeline:

1. Label four rigid machine features in the reference image and each inspection image.
2. Register the inspection image to the reference image.
3. Crop the configured connector regions at reference-image resolution.
4. Return `uncertain` until a connector-state model has been trained on labelled crops.

## Commands

```powershell
python tools/label_anchors.py data/raw/图片1.png --output data/labels/图片1.anchors.json
python tools/label_anchors.py data/raw/图片2.png --output data/labels/图片2.anchors.json
python tools/register_and_crop.py --config config/machine.json --image data/raw/图片2.png --anchors data/labels/图片2.anchors.json --output output/图片2
```

After anchor labels are present, add each connector ROI to `config/machine.json` using reference-image pixels:

```json
"J1": {"left": 100, "top": 200, "right": 300, "bottom": 400}
```

## Current visual-review prototype

The operator-facing entry point is `启动装配自动定位复核_DINO融合版.bat`.
It performs full-frame SIFT/MAGSAC homography alignment, rejects unreliable
alignment before producing candidate boxes, then combines conventional visual
difference with DINOv2 feature difference inside configured review regions.
Its output is a human-review prompt, not a component-level fault diagnosis.

## Zhijie inspection Agent core

`inspection_agent/` adds an auditable task workflow around the existing visual
report without changing its evidence boundary.  It records tool calls, keeps
machine candidates separate from human conclusions, compares a deliberately
limited declared connection graph, requires human confirmation before repair
guidance, and supports reinspection history.  The PyQt operator interface now
creates `agent_task.json` beside the visual report and exposes explicit human
review and repair-guidance controls.  A corrected or recaptured image can be
run through the same detection button and appended to that task as a
reinspection.  The command-line entry point is `tools/inspection_agent_cli.py`.
See `inspection_agent/README.md` for commands and the synthetic topology
contract examples.

The SAM3-to-topology proof of concept is now runnable through
`tools/sam3_topology_adapter.py`.  It accepts the existing per-mask visible
endpoint report plus operator-declared terminal ROIs.  A graph edge is emitted
only when both endpoints of one independent visible segment map uniquely to
two different ROIs.  SAM3 confidence is preserved, so low-confidence evidence
remains `insufficient_evidence` rather than becoming an automatic fault.

For the fixed Mendeley chassis dataset, `tools/evaluate_mendeley_normal_bank.py`
adds an image-level review gate backed only by `train01` normal examples.  It
selects a threshold on `val01`, freezes it for `test01`, and writes a reusable
`normal_bank_model.npz`.  `tools/classify_mendeley_normal_bank.py` applies that
artifact to one image.  This dataset-domain gate does not replace candidate
localization and is not evidence of industrial field accuracy.

### Regression suite

`config/regression_suite.json` references the original desktop sample photos;
it deliberately does not move or rename them.  A fixture can expect either a
possible-difference prompt or the safe alignment-uncertain result when the two
photos cannot be reliably registered.

Run the headless suite after changing alignment, DINO fusion, or thresholds:

```powershell
.\.venv\Scripts\python.exe -B prototype\evaluate_regression_suite.py
```

The JSON result is saved under `output/regression/`.  On this Windows setup,
PyTorch must be imported before any module that imports PyQt; the DINO launch
entry and regression runner preserve that order.

### Local model assets

The DINOv2 source and ViT-S/14 checkpoint are provisioned under `models/dinov2/`.
The runtime loads only this local copy; it does not fall back to a user cache or
network download.  `models/LightGlue/` contains the official LightGlue source,
prepared for a later alignment A/B test but not yet enabled in the inspection
pipeline.  Both model directories are intentionally ignored by Git because
they are third-party downloadable assets.
# 最新代码同步（2026-10-09）

本次补充连接补证与遮挡研究源码，以及可选的大模型逐框判断、受本地证据约束的分级复核和Qt展示桥接。正式视觉入口保持原已验收版本；新增功能通过独立入口运行。详见 [研究模块与使用条件](research/STATUS_20261009.md)。经用户授权的机柜试用照片现已公开；接口密钥和无关私人取证结果不公开，既有SAM模型Release保持不变。

## 可控思考与比赛试用

2026-10-09启动修复：选择两张有效照片即可开始；未指定检查区域时自动检查全图，已有自定义检查区域仍保留。检测阶段与拍摄建议显示在窗口中；拍摄建议由本地规则生成，无需启用云端模型。

运行根目录`启动大模型复核版.bat`，勾选“启用DeepSeek辅助复核”后才显示设置。支持关闭、轻量、标准、深度四档思考，默认推荐轻量；支持本机密钥记忆和纯掩膜/脱敏局部图选择。外发前仍须检查预览，本地检测结果完整保留。

- [比赛资料、部署说明与答辩课件](competition/submission_materials_20261009/README.md)
- [评委试用机柜照片与操作方法](examples/cabinets/README.md)

密钥保存在本机忽略发布的配置文件中，不随报告、案例或源码上传。界面与传参已通过离线回归；单例云端推理记录不是跨机柜准确率认证。

