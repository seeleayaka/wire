# 正式Agent工单：局部证据附加入口

新增 local_evidence_bridge 和 InspectionTask.record_local_evidence_review。原工单没有local_evidence_reviews字段也可导入，旧状态机语义不变。

入口 tools/import_local_evidence_review.py 参数：--task 现有工单 --bundle 元数据包 --review 页面导出记录 --evidence-files 来源文件清单 --case 选定照片，可附 --plan 预期表草稿。

来源清单形如 {"cabinet_1":{"map":"地图文件","endpoints":"端点报告","audit":"路径报告"},...}，相对路径以清单所在目录解析。必须包含证据包全部照片，且端子/候选/ROI与来源一致；工单待检照片必须匹配选定case。

默认拒绝automation_fixture。--allow-test-records仅明确测试回放使用，记录类型保留为测试，绝不伪装人工确认。署名是自声明而非认证。

工单必须awaiting_human_review。导入后保持该状态，只追加local_evidence_reviews、工具调用和历史，不写human_conclusions，不给出维修指令，不生成连接边或执行拓扑比较。预期表永远作为未确认草稿附加。真正的人工故障确认继续使用原record_human_review流程，局部支持候选ID不能直接当其确认ID。

核验元数据包编号、当前原图/地图/端点/路径SHA、端子范围及候选来源。保存前核对工单未变化，临时文件写完后同目录替换；失败前保持原内容。同份意见在同一视觉分析轮次重复导入拒绝；复检的新轮次单独记账。此非并发数据库或签名审计系统。

验证：7新增桥接测试、12工单+2GUI契约回归通过。两份真实页面下载的测试记录经正式CLI导入存档证据回放工单，默认测试拒绝/显式测试导入/保存重载/重复拒绝通过。存档回放未重跑视觉，不是全链GUI验收或准确率证明。

本轮未添加GUI导入按钮，未改原视觉/SAM推理或拓扑比较算法。下一步把入口接入主界面并验证正确照片和过期工单回退。
