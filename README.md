# 智接 / WireMind — 电气装配复核 Agent

当前项目和全部主要模型（含 3.45 GB SAM3）的备份见
[恢复说明](backup/README.md) 与
[模型 Release](https://github.com/seeleayaka/wire/releases/tag/backup-20261004)。
本仓库是公开的脱敏快照，不含密钥或私人样本。
已验收主线与新候选算法分开保存；新模型不会自动启用。

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
