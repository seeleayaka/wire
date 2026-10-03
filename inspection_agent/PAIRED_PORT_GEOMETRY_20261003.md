# 配对定位补漏：默认关闭增强路径

已通过固定源验收、真实源/正常参考/ROI门控和一次新的完整SAM/Agent/Qt按钮流程。
沿用“增强补漏”第四开关，四个开关默认均关闭；不新增开关，不替换原提示。
模型只增加需人工复核的端口线索，不确认物理断接、电气连通性或现场故障。

## 实验结果和边界

| 来源 | 原V3命中/目标 | 固定新全量头 | 未匹配提示 |
|---|---:|---:|---:|
| TRAIN192 | 277/344 | 286/344 | 4 → 4 |
| 内层48 | 63/80 | 64/80 | 0 → 0 |
| 外层30 | 33/56 | 33/56 | 1 → 1 |

这些是现有、被反复使用的同相机场景数据；检测器训练过全部TRAIN192。
新分类头三折OOF训练源为281/344，而运行全量头为286/344，不混用计数。
不是独立现场、跨机柜或电气故障准确率。旧框不删除，原5主要+5附加额度不增。

训练只加统一中心偏移和尺寸负例，按全部原同源GT重新生成训练类别，不改原标签。
冻结DINO，正常参考映射回待检坐标，比较1.5/3倍上下文；使用固定.98分类门槛、
.85上下文有效覆盖及原参考否决、ROI、warp.98保护。无文件名/照片坐标特例。
遇到局部ECC坐标校正，原桥明确弃权；零提示不等于模型识别成功。

## 实际完整链路

真实诊断15个候选/风险来源通过，另3个正常控制新跑后均安全弃权，旧输入和已完成
输出SHA核验通过。新训练9个、内层1个正确提示均通过实际参考/ROI门控，没有新增
未匹配。原18场景测试的applied-only断言故障已保留并用独立契约测试验证，不修改
保护去强行通过。

新inspection SAM推理、SHA核验的normal073参考SAM缓存复制、原最终回调、Agent任务、
实际Qt增强按钮完整通过514.76秒；选择首个诊断有增益的TRAIN场景，7/10→8/10、0
未匹配。实际叠图已查看。SAM未完成时按钮不可用；头变化回调拒绝写入；原候选及
提示前缀保持。最终possible_difference_manual_review/awaiting_human_review。

工作区证据：

- artifacts/paired_geometry_live_resume_20261003/report.json
- artifacts/paired_geometry_complete_sam_20261003/acceptance.json
- artifacts/paired_geometry_project_regression_20261003/report.json（366测试通过）
- artifacts/paired_geometry_formalization_parity_20261003/report.json（270源提议/选择一致）

运行头output/paired_port_geometry_20261003/last_head.pt SHA238f517b…，
config/paired_port_geometry_20261003.json SHA2b716d54…；features/encoder/原运行指纹
绑定，漂移则保留原V3明确回退。Manifest保存接入前来源证明，SAMpending字段是该
预注册快照，不是实时任务状态；实时完整链路以acceptance.json为准。
原教师/学生/特征头/1280分支/校准的SHA保持。所有后续实验保持在工作区，不替换
这条已验收路径，直到同等级泛用源门控和实际完整流程通过。
