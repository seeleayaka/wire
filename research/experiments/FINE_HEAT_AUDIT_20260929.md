# 细网格热峰候选：整批对照结果

2026-09-29。结论：单纯加密特征网格没有解决本固定候选规则的主要漏检。局部损伤框有少量贴合改善，但覆盖、整图及各类指标仍未达到正式基线。正式算法未改，不继续扫描分辨率或阈值。

## 已执行的固定协议

协议FINE_HEAT_PROTOCOL_20260929.md在读取验证计分前保存。完全沿用相同80拟合/10间隔/30校准正常图和32通道，等权位置均值在细网格重算；局部k=3、邻域1格变2格，保留近似实际120像素容忍距离。只读已有784输入41x56x384缓存，没有下载权重、新数据或重新提取DINO。

复用同一个probe_heat_candidates.py：3x3极大值、完整等高平台检查、8邻接、半峰值生长、原每图候选上限；不足不补框。3x3的实际像素范围随网格变化，本轮没有另改窗口。正常校准规则不变，具体阈值由同30正常图重新计算：粗5.48975，细8.08205。不能把这称为同一个数值阈值；细网格增加格点，也会改变正常极值分布。

全部30张val01原基线候选字典精确重放，test01未使用。2项新均值测试、4项候选单测和12项空间测试通过，共18项。prepare_fine_heat_evidence.py按图累计归一后的32维均值，避免一次分配整个80张384维float64数组；兼容读取器的单位精度矩阵不是重新拟合的协方差。

## 粗/细网格结果

| 策略 | 来源框交集 /245 | 故障图交集 /15 | 图片等权覆盖 | IoU>=0.1 | IoU>=0.5 | 平均最佳IoU | 故障候选 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原正式基线 |146|15|65.44%|23|4|0.03267|70|
| 粗网格仅新框 |46|12|28.23%|34|2|0.03780|64|
| 细网格仅新框 |47|12|29.55%|29|3|0.03842|58|
| 粗网格首框＋新框 |109|12|33.90%|31|2|0.03366|68|
| 细网格首框＋新框 |109|13|37.06%|29|2|0.03682|64|

仅新框三类覆盖：粗9/143、31/83、6/19；细11/143、29/83、7/19。细网格损伤出现1个IoU>=0.5的来源框交集（粗0），损伤平均最佳IoU由0.00743到0.01695，但来源框覆盖仍仅11/143。不能凭一个精确交集或小幅总分提升升级。

保留首框后，细网格三类覆盖75/143、27/83、7/19；原基线81/143、51/83、14/19。故障图少漏一张仍没有恢复15/15，图片等权和类别覆盖明显退步。新方法并未取得净提升。

## 生成阶段与正常响应

粗网格截断前81个故障候选覆盖55/245、12/15图；细网格截断前79个覆盖55/245、13/15图，精确交集由2到3。更多格点并没有带来更多有效覆盖；本固定规则的主要遗漏仍在生成阶段，而不是增加候选数量就能恢复。

正常强制定位：粗仅新框0候选，细仅新框1候选/1张图；首框＋新框粗细均15候选，原基线37。完整级联正常门未改，这些不是现场或整套系统误报率。

本对照排除了“只要在此规则中提高分辨率就能解决主要漏检”的简单解释，不证明所有细特征、CNN纹理或其他候选方法无效。阈值选择、语义特征的局部敏感性和峰值生成假设仍有混合影响，不能把结果完全归因为模型没有损伤信息。

## 产物与复现

tools/prepare_fine_heat_evidence.py、tests/test_fine_heat_evidence.py。

output/mendeley_fine_heat_evidence_20260929/original保存正常成员清单、缓存SHA256、细网格均值模型及30张校准图；metric保存30张验证融合热图与正常尺度。output/mendeley_fine_heat_candidates_20260929/report.json保存正常最大值、逐图生成框、截断前后及各类计分。旧缓存不含模型版本指纹的限制仍保留，不因记录文件哈希而视为彻底解决。

```powershell
.\.venv\Scripts\python.exe -B tools\prepare_fine_heat_evidence.py --coarse-maps output\mendeley_spatial_normal_20260928 --feature-cache output\mendeley_local_normal_bank_highres_20260928 --fault-cache output\mendeley_merge_audit_20260928 --normal-cache output\mendeley_normal_evidence_20260928 --output output\mendeley_fine_heat_evidence_20260929
.\.venv\Scripts\python.exe -B tools\probe_heat_candidates.py --original-maps output\mendeley_fine_heat_evidence_20260929\original --metric-maps output\mendeley_fine_heat_evidence_20260929\metric --feature-cache output\mendeley_local_normal_bank_highres_20260928 --fault-cache output\mendeley_merge_audit_20260928 --normal-cache output\mendeley_normal_evidence_20260928 --output output\mendeley_fine_heat_candidates_20260929
```

复跑换新的两个输出目录，不覆盖记录。用户已有7个脏跟踪文件、数据和标注未动，全部结果只属于隔离开发期评价。

## 下一项：独立CNN纹理证据

已只读核验工程虚拟环境元数据：torch2.13.0、torchvision0.28.0、ultralytics8.4.121；常用torch权重缓存仍只有dinov2_vits14_pretrain.pth。models中已有yolov8s-seg.pt及wire_harness_yolov8s_seg_best.pt。依赖已安装不等于CNN特征分支可运行或已验证；尚未加载新CNN、提取特征、训练或对照片计分。

下一安全入口是核验可复用的预训练卷积中间层与权重来源，然后做同正常成员的纹理分支对照。若复用YOLO backbone，只能称为CNN纹理原型，不是PatchCore正式复现；若选ResNet/WRN，需核验权重、许可证及隔离缓存后再运行。不要同时换候选规则、模型和阈值追val01。

走线仍需要对象关系证据，SAM对应方向尚未实施。上述局部异常方法不等于电缆身份、端子归属、电气通断或现场故障分类。
