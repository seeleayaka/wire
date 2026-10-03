# Mendeley 合并前分组试验（2026-09-28）

结论：11 个变体均未达到升级条件，保留正式算法。没有修改生产分组、候选预算或 test01 参数。

## 方法与可复现性

固定 train01/normal_073.JPG 作为参考，使用 val01 的 15 张故障图、245 个已对齐标注框、候选上限 6。先真实运行现有 DINO 流程，保存合并前观察框；15 张图的最终框坐标和顺序均与原报告一致。离线重放进一步验证原算法全部局部候选字典完全一致。

实验仅替换分组规则，其余证据聚合、排序、发布筛选和预算计算沿用原代码。试验包括去掉近邻连接、限制间接连锁、组内全配对、限制合并面积扩张，以及逐图不超过原候选数的版本。匹配数量上限不使用标注。标注仅用于事后评价。

## 结果

| 分组模式 | 标注框有交集 /245 | IoU>=0.5 /245 | 平均最佳 IoU | 候选总数 |
|---|---:|---:|---:|---:|
| baseline |146|4|0.03267|70|
| overlap_only |116|2|0.02817|80|
| seed |118|4|0.03024|84|
| complete |93|3|0.03145|86|
| growth_1.5 |115|5|0.03785|82|
| growth_2 |124|5|0.03807|81|
| growth_3 |136|4|0.03587|75|
| growth_4 |143|4|0.03432|71|
| growth_6 |145|4|0.03348|71|
| growth_2_matched |109|4|0.03278|70|
| growth_3_matched |131|4|0.03382|70|
| seed_matched |111|2|0.02403|70|

所有模式在图像级都有至少一个框碰到标注（15/15），但这不等于找到全部故障，更不等于现场准确率。IoU 是框位置的贴合程度。“有交集”是宽松覆盖指标，不代表正确故障分类。

限制合并可改善部分框的贴合度，但也拆散多尺度证据，改变后续排名和动态候选预算。最温和的 growth_6 仍少覆盖 1 个标注、多显示 1 个框。固定候选数量也没有解决漏检增加的问题。没有模式达到标注覆盖不下降、候选不增加的基本条件，因此没有继续跑正常图或 test01，也没有升级到正式路径。

三个单元测试覆盖连锁分组、面积扩张限制和缓存不被变体修改。完整实测与逐图数据见 output/mendeley_merge_audit_20260928/comparison.json。

## 复现命令（项目根目录）

```powershell
.\.venv\Scripts\python.exe -B tools\merge_audit.py capture --output output\mendeley_merge_audit_20260928
.\.venv\Scripts\python.exe -B tools\merge_audit.py replay --output output\mendeley_merge_audit_20260928
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_merge_audit.py -v
```

capture 依赖现有数据集、模型和原 val_fixed_faults_budget6.json；重放核对 tiled_dino_review.py 源码哈希。缓存不是通用独立数据集。

## 后续入口

停止单纯几何拆框的参数搜索。若继续精度研究，下一项候选是检查合并前局部证据如何区分正常变化和故障；先做有限样本诊断，再决定是否实现，不宣称已经证明有效。按既定时间限制推进人工确认—复检—报告的完整演示，保持 possible_difference_manual_review 能力边界。
