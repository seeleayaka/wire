# 新跑SAM来源绑定与拓扑检查（2026-10-01）

来源绑定/端点提取链路完成；真实端子接线识别未成功。本轮未训练模型、未改GUI、未改变正式视觉流程或分数门槛。

## 实际执行

固定使用原始机柜1/1.png，全幅620×622，复制字节相同的源图到新目录 source_snapshot.png。原始拍摄者来源未独立核验；不用AI编辑 wrong2。新输出目录 cabinet_1_fresh，运行前不存在，无图像状态缓存可复用。

现有本地SAM3、原运行脚本、提示 cable、掩膜门槛0.5、CPU8线程、内部输入分辨率1008。门槛沿用既有端点探测默认，不改变正式GUI融合的0.4配置。原图无配准、裁切、缩放；模型内部预处理与输出还原由原脚本处理。

SAM报告12个实例掩膜；现有端点提取器按掩膜、独立连通分量处理，50像素门槛，14个分量，11个无分支两端几何候选、3个几何弃权。模型记录耗时：构建9.672秒、图像编码231.1秒、文本/掩膜17.902秒；这不是GUI整链耗时。

源图、快照、原推理脚本、提取器、几何函数、端子地图模块、SAM权重完整SHA256核对运行前后不变。各掩膜、SAM报告和JPEG预览绑定本轮指纹；预览JPEG是展示副本，不冒充源图字节身份。坐标系明确为未配准源图像素。

新增来源绑定模块只接收新跑、输入输出核验后的manifest，拒绝来源不明的旧缓存、不明图像状态复用、掩膜尺寸/指纹、实例数量、记录掩膜ID与分数不一致。绑定不是电气关系认证；调用方能伪造JSON声明，软件核验也不能证明旧缓存历史身份。这里的来源证据来自本轮实际执行与独立重算，而不是仅填写字段。

## 核查结果

- 独立从本轮12张掩膜重算14个分量与全部几何记录，精确一致；拓扑入口复核输出一致。
- 原始照片、端点叠图、14分量审计图均查看。掩膜主要覆盖前景粗线和图边竖直长物体，后者类别仍需人工核查；没有足够端子排细接线证据。
- 三处孔位草稿与已有端点纯坐标接触数均0。不移动孔位救匹配、不减门槛、不合并片段推断隐藏路径。三个孔位仍未确认，不发连接边。
- 实际 topology_review 为 insufficient_evidence，observed_graph=null、raw_comparison=null。缺孔位确认、预期表与观察覆盖，不能叫漏接/断线或接线正确。
- 来源绑定8项测试通过；构造测试仅验证契约，不算现场精度。重算一致仅证明提取复现，不证明SAM掩膜语义正确。

## 原项目入口

新增 inspection_agent/sam_endpoint_binding.py、tools/run_bound_sam_endpoints.py、tests/test_sam_endpoint_binding.py 和说明 BOUND_SAM_TOPOLOGY_20261001.md。独立显式调用，不改正在使用的SAM原脚本、历史缓存或端点提取器。

用法：`.venv/Scripts/python.exe tools/run_bound_sam_endpoints.py --image IMAGE --map PORT_MAP.json --output NEW_RUN_DIRECTORY`。map可省略；输出目录必须新建，旧运行拒绝覆盖。只支持不经配准的源图像素；配准图必须另有实际坐标变换证据，不能自动改字段套用。当前平台配置为Windows本地SAM运行时。

产物：cabinet_1_fresh/run_manifest.json、sam/report.json、bound_endpoint_report.json、topology_review.json、verification.json、draft_ports_and_fresh_endpoints.png、endpoints/visible_segment_audit_sheet.jpg。

下一拓扑改进首先是细接线的可见证据覆盖，不是放宽图比较条件：采用固定方案对端子/细线区域做检测覆盖审计，与全幅结果对照；真实孔位、线号和两端身份需继续核对。即使细线被检测到，端点进线槽仍不能补成端子到端子的连接，预期接线表与隐藏路线需人工补证。不能只靠本张图选择提示/阈值并声称泛用性提高。
