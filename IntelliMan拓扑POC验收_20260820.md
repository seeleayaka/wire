# IntelliMan T5-3 拓扑 POC 验收

日期：2026-08-20

## 结论

IntelliMan T5-3 已在本机 CPU 隔离环境中跑通，能将一张本地二值线束掩膜
转换为 `topology.json` 和节点/边叠加图。该结果只证明离线拓扑工具链可运行、
结构化输出完整且同输入可复现；尚未验证线缆柜跨域适配，不得接入 UI、规则引擎
或生产 NG 判断。

## 范围与边界

- 原始来源：`E:\edge\INTELLIMAN.zip`，SHA-256 为
  `56B7C27075ECE486D95F36619CF4667411DA00204CEC558CD8DB8E018A851766`。
- 原压缩包未修改，仅解压至 `E:\airecognizewire\third_party`。
- 运行入口：`E:\airecognizewire\tools\run_intelliman_topology.py`。
- 输入：一张本地 8-bit 二值 Cable mask；官方验证样图为 640 x 480。
- 输出：`topology.json`、`topology_overlay.png`，字段包括节点、边、路径、
  几何交点、分支点和局部 branch score。
- 两份 checkpoint 仅以 `torch.load(weights_only=True)` 加载；本轮没有读取
  IntelliMan 提供的 `.pickle` 数据样本。

## 验收结果

| 项目 | 结果 |
| --- | --- |
| 两份训练权重加载 | 通过，CPU 模式 |
| 官方二值样图 | 20 / 20 成功，20 / 20 JSON 结构校验通过 |
| 图规模 | 40-49 节点，2-3 条路径 |
| 几何事件 | 合计 7 个交点、23 个分支点 |
| 可复现性 | 样图 01 重跑后，输入、模型、拓扑、汇总和警告字段逐字一致 |
| 单张耗时 | 单进程冷启动约 8-10 秒，模型不常驻 |

结果目录：

- `E:\airecognizewire\outputs\intelliman_t5_3_official_regression`
- `E:\airecognizewire\outputs\intelliman_t5_3_sample_01`
- `E:\airecognizewire\outputs\intelliman_t5_3_repeat_01`

## 人工复核与限制

人工查看样图 01、02、11、18 的叠加图：主线节点和连边总体贴合掩膜中心线；
但样图 11、18 中的孤立白色前景会保留为孤立图节点。由此可知：IntelliMan
不能承担原始图像分割，也不能自动忽略标签、夹具、背景或碎片。

当前环境没有适配 PyTorch 2.13 CPU 的 PyG `pyg-lib` / `torch-sparse` 二进制
轮子。启动器以原生 PyTorch CPU 实现最远点采样和 KNN 建图，并为未使用的
`torch_sparse.SparseTensor` 导入提供最小兼容层。该实现已足以跑通本 POC，
但尚未与官方编译算子逐输出比对，因此不能据此声称模型精度完全等同于论文环境。

## 后续质量门

等待本地 YOLOv8-seg 训练可用后，再执行以下顺序：

1. 输出线缆柜的 Cable 二值 mask。
2. 去除非线束前景、孤立小连通域和不连续伪掩膜。
3. 固定掩膜尺度与预处理，建立正常图重复性回归。
4. 对已知错误图评估断线、分支、路径变化的检测价值。
5. 只有以上指标稳定，才评估是否将 `topology.json` 作为 DINO 人工复核的辅助证据。
