# 可复用端子映射与拓扑入口（2026-10-01）

完成独立可复用入口，并接回原 `connection_graph_from_sam_endpoints` 和 `compare_topologies`。这是软件串联和证据边界测试完成，不是现场自动接线识别完成。原视觉主流程、GUI、SAM模型与阈值不改，无重新训练/推理。

## 新入口做什么

- 对任意可读取的源图建立图像指纹、尺寸、源图像素坐标绑定的端子地图草稿。可导入上轮对象清单作为 body_objects，绝不自动将外壳框改成接线孔。
- 每个接线孔单独记录设备ID、端子标签、ROI、确认状态、复核人和证据说明；线号可留空。不同柜子更换地图，算法不含机柜专用坐标/编号。改变图像或配准坐标必须另建/核查地图，不自动套用。
- 预期连接表 null 表示未知，空数组表示明确声明没有连接，二者不混淆。独立保留预期表复核与限定范围完整性。
- 端点报告须绑定同一图像/坐标系，带现有几何契约v1。未确认孔位不产生候选边；不允许“按最近距离猜端子”。
- 观察范围完整性须另有复核记录，不能从SAM没有掩膜推断。空端点结果、未知期望表、未确认孔位、分支/重叠/未归属端点均保留证据不足。
- 固定沿用0.75分数门槛。候选关系满足显式条件后才调用原比较器。原比较器结果留作 raw_comparison，外层最终结论保留缺证据保护；原比较器自身语义未改变。
- 即使声明关系一致，也只输出 declared_visible_topology_agreement_manual_review，不输出“接线正确/电气连通/自动故障诊断”。

## 本轮检验

16项新测试通过。包括真实几何函数计算的构造线段→端点适配器→图比较器，以及改编号、改场景、同比放大坐标的新构造场景，无修改算法。构造场景仅测试契约和不写死坐标，不作为跨机柜精度证据。

固定核查6张已有照片（端子排001/002/003首视角，原始机柜1/2/5）。端子排66个对象包括45个外壳、16附件、5桥接件，未自动生成孔位。机柜1给出3个下方接线入口粗ROI草稿，放大图已查看，但身份/精确孔位仍未确认；机柜2/5保留空地图等待填写，没有编造端子/连接。6张均为 insufficient_evidence，未执行SAM、未输入虚构端点。

原始机柜1/2/5可能是同一机柜的不同视角，不能当三台独立机柜。原始拍摄者来源未独立核验。先前AI编辑 wrong2 素材不参加本轮真实素材核查。

## 使用与接续

项目新增：inspection_agent/terminal_mapping.py、tools/run_terminal_mapping.py、tests/test_terminal_mapping.py。入口是显式调用，GUI暂未增加编辑器。

创建：`.venv/Scripts/python.exe tools/run_terminal_mapping.py draft --image IMAGE --scene-type SCENE --output NEW_MAP.json`，可加 `--inventory OBJECT_INVENTORY.json`。

复核：`.venv/Scripts/python.exe tools/run_terminal_mapping.py review --image IMAGE --map MAP.json --output NEW_REVIEW.json`，可加 `--endpoints BOUND_ENDPOINT_REPORT.json`。输出已存在则拒绝覆盖。

每个 port 字段示例：id="X1:1:lower"、device_id="X1"、terminal_label="1 lower"、roi_kind="wire_entry_port"、bbox_xyxy=[x1,y1,x2,y2]、confirmed=false、reviewer=null、evidence_note="待核对"。只有确实人工核查后才能改 confirmed=true，并填写 reviewer/evidence_note。连接表只能来自可追溯接线资料或人工核验，不能从期望结论倒推。

注意：当前历史SAM报告通常没有 image_binding/coverage_review。新入口不会自动给旧缓存补来源身份，不直接接受坐标来源不明的报告。生成端点的一侧需要保存实际源图/配准坐标指纹；字段声明本身也不能追溯证明旧缓存身份。下一拓扑工作应是给实际源图的端点生成流程补上可追溯坐标绑定，再用已确认孔位验证两端归属；仍缺的预期接线表与隐藏路线必须人工补证。

映射入口可以复用到其他柜子；真实跨机柜效果尚未验证。原用户已有改动保留，不提交Git、不降低阈值救结果。
