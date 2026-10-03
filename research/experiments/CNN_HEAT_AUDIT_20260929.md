# CNN纹理证据：生成失败，旧池排序有进展但尚不能升级

2026-09-29。原正式算法未变。本轮完成离线CNN特征、正常建模/校准、整批候选生成、坐标核对及一个明确后验的旧池排序对照。CNN热峰新框整体失败；CNN给原合并池排序取得开发集覆盖提升，但走线类退步，而且选入了原资格门会过滤的框，因此保留实验，不直接上线。

## 实施及数据边界

冻结协议见CNN_HEAT_PROTOCOL_20260929.md。复用已有通用models/yolov8s-seg.pt，声明coco.yaml/80类，不用旧线束微调模型，不下载、不训练。权重SHA256为0bac0770b55e5eb5b76a61bc535673288dcec36c2bc0cd25ee0d584c632f3413。它是通用YOLO前缀卷积原型，不是PatchCore复现，不能沿用作者基准准确率。

索引2/4卷积层、3x3平均、21x28网格、分层L2归一后拼接192通道；最大输入边392，固定种子选32通道。正常80拟合/10间隔/30校准成员不变，局部3近邻/1格容忍及等权位置均值分支正常尺度校准后取最大。140次新配准全部通过质量门；保存图像、参考、权重、提取代码、配准代码和库版本指纹。val01为15故障＋15正常，test01本轮未读。

新CNN和旧DINO的架构、预处理与具体提取流程不同，不是严格只改一个张量的对照。15张故障图以本轮实际生产配准函数捕获的H变换原始标注，与旧标注坐标15/15完全一致；审计16张配准质量报告也与提取时完全一致。四类固定最小编号面板已全部查看；CNN top5%热点无无效warp格。本核对不恢复旧缓存的实际H，也不证明局部非刚体线束完全对齐。

## 冻结热峰规则的结果

正常每图最大值p95阈值3.5980513，同3x3峰值、完整平台检查、8邻接半峰生长、原每图上限；不足不补框。30张原候选字典精确重放。

| 策略 | 来源框交集/245 | 故障图交集/15 | 图片等权覆盖 | IoU>=0.1 | IoU>=0.5 | 平均最佳IoU | 故障候选 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原正式基线 |146|15|65.44%|23|4|0.03267|70|
| CNN仅新框 |74|14|30.87%|26|1|0.02793|39|
| 原首框＋CNN新框 |106|14|34.95%|27|1|0.02970|51|

仅新框三类覆盖45/143、22/83、7/19；首框＋新框77/143、22/83、7/19，均低于原81/143、51/83、14/19。两种都漏damaged_012.JPG。截断前42个候选覆盖82/245、14/15图、精确交集3；主要遗漏在生成阶段已发生，不能只增加最终上限解决。

正常强制定位仅新框0、首框＋新框15，原37；完整系统正常门未改，不能称系统或现场误报率改善。该固定生成方案不保留为正式路径。

## 后验诊断与一次固定旧池排序

按来源框相交粗格点计算的图片等权排序诊断，融合DINO0.72493、CNN0.85235；损伤0.74586→0.89803、断接0.70045→0.80950、走线0.72848→0.84952。等权位置分支0.71945→0.80236。来源标注碎片化且不完整，这不是缺陷像素AUROC，更不是电缆身份识别；支持的是“相对局部响应更有用，但热峰输出损失覆盖”的开发期假设。

在看排序计分前冻结一个后验实验：旧首框不动，用相同anchored_selection及region_score p95在同一旧合并池排序剩余槽位，数量严格等于原算法。CNN融合图对照DINO等权融合图及原基线；不生成新几何、不改阈值/层/分位。旧池在资格过滤前，与此前DINO锚定实验一致。DINO控制候选与旧报告30/30完全一致；加面积诊断后在新目录复算，全部策略结果与第一次完全一致。

| 策略 | 来源框交集/245 | 故障图交集/15 | 图片等权覆盖 | IoU>=0.1 | IoU>=0.5 | 平均最佳IoU | 故障候选 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原正式基线 |146|15|65.44%|23|4|0.03267|70|
| 同池DINO排序 |150|14|54.41%|24|4|0.03695|70|
| 同池CNN排序 |190|15|73.41%|29|4|0.04387|70|

CNN三类覆盖123/143、54/83、13/19，原81/143、51/83、14/19；按图7张改善、3张退步、5张不变。走线损失集中于misrouted_040.JPG，断接disconnected_002/027也有覆盖下降；不能据这些文件写特判。精确交集总数仍4，分布由断接4/走线0变断接3/走线1，损伤仍0。正常候选仍37，9/15张选框几何改变。

故障候选并集平均ROI覆盖面积从20.30%降到16.57%，正常从20.19%到18.61%；覆盖提升不是整批平均框面积膨胀造成，但也不是每张都更小的证明。另一个关键限制：CNN选中30/70个故障框、10/37个正常框不在原展示资格池；同池DINO为37和9。原过滤门主要依赖既有证据，CNN是否能为这些框提供可靠替代资格尚未验证。这是更好的实验排序，不是保持所有生产过滤规则的正式升级。

## 判断与下一安全入口

1. 保留冻结CNN特征及旧池排序原型；放弃这次固定热峰生成配置。没有回写正式GUI、正常门、SAM3或数据/标注。
2. 下一项先把“CNN有效证据”与候选资格关联：整批审计原资格门为什么过滤新增选中框，使用正常训练/校准构建通用证据资格或融合边界。必须报告走线/断接回退、正常负担、面积与IoU，不从几个文件写规则；不扫描热峰阈值追val01。
3. 冻结新策略后才做独立/分组验证。当前反复使用val01且同机箱，无法据此宣称泛化或现场准确率；不要用test01挑规则。若没有真正分组新场景，明确报告该限制。
4. SAM对象对应仍未实施，走线可后续加入对象关系证据；候选系统仍为possible_difference_manual_review，不等于电缆身份、端子归属、电气通断或完整拓扑。

## 可复现产物

tools/prepare_cnn_heat_evidence.py、audit_cnn_heat_evidence.py、probe_cnn_candidate_rank.py及2份测试文件。12项空间、8项热图、2项面积、3项锚定测试，共25项实际通过；面积单测覆盖重叠/包含/重复/分离和非法坐标。

output/mendeley_cnn_heat_evidence_20260929保存140个特征缓存、正常成员、校准图和30张验证图；mendeley_cnn_heat_candidates_20260929保存热峰结果；mendeley_cnn_heat_audit_20260929保存坐标及4张面板；mendeley_cnn_candidate_rank_20260929保存首次排序；mendeley_cnn_candidate_rank_verified_20260929保存相同结果和并集面积/资格审计。所有目录均新建，不覆盖旧实验。

```powershell
.\.venv\Scripts\python.exe -B tools\prepare_cnn_heat_evidence.py --coarse-maps output\mendeley_spatial_normal_20260928 --fault-cache output\mendeley_merge_audit_20260928 --normal-cache output\mendeley_normal_evidence_20260928 --weights models\yolov8s-seg.pt --output output\mendeley_cnn_heat_evidence_20260929
.\.venv\Scripts\python.exe -B tools\probe_heat_candidates.py --original-maps output\mendeley_cnn_heat_evidence_20260929\original --metric-maps output\mendeley_cnn_heat_evidence_20260929\metric --feature-cache output\mendeley_cnn_heat_evidence_20260929\features --fault-cache output\mendeley_merge_audit_20260928 --normal-cache output\mendeley_normal_evidence_20260928 --output output\mendeley_cnn_heat_candidates_20260929
.\.venv\Scripts\python.exe -B tools\probe_cnn_candidate_rank.py --cnn-maps output\mendeley_cnn_heat_evidence_20260929\metric --dino-maps output\mendeley_spatial_metric_20260929 --fault-cache output\mendeley_merge_audit_20260928 --normal-cache output\mendeley_normal_evidence_20260928 --output output\mendeley_cnn_candidate_rank_verified_20260929
```

复跑使用全新输出目录。仅提交本轮隔离工具、测试、协议与审计，不纳入用户7个既有脏跟踪文件、模型、数据、缓存或凭据。
