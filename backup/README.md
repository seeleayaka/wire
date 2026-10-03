# 当前项目备份：2026-10-04

仓库保存代码、网页/窗口界面、测试、配置、第三方源码/许可及研究脚本。
完整 SAM3 权重和其余 37 个模型文件在同一仓库的 `backup-20261004` Release 中。
SAM3 是 3,450,062,241 字节，拆成 4 片；不是占位文件或只给下载来源。
所有模型均记录 SHA-256，恢复后会校验完整 SAM 文件。

## 恢复

在克隆仓库后，用 Python 3.11 运行：

```powershell
python backup/restore_models.py --download
```

也可手动下载 Release 的模型 ZIP 和四个 SAM 分片到一个目录，再运行：

```powershell
python backup/restore_models.py --assets "D:/downloads/wire"
```

需要额外约 8 GB 空间保存下载件及恢复后的模型。脚本不覆盖不同内容的旧模型。
本机 Python 环境没有打包；安装依赖用现有 `runtime/bootstrap.ps1`（Windows）
或 `runtime/bootstrap.sh`（Linux）。端口补漏研究另需
`pip install -r backup/requirements-research.txt`。SAM 独立环境和主环境不得混装。
这些依赖版本来自本机快照；本次没有在全新电脑验证重新安装。

操作入口：`启动装配自动定位复核_DINO融合版.bat`；Linux 用 `start_cable_review.sh`。
需自行选择标准图和待检图。没有公开私人机柜照片、原始数据集、密钥、
本机设置、虚拟环境、缓存、历史 Git 数据及旧发布压缩包。
因此这是可恢复的代码和模型备份，不是包含全部原始照片的磁盘镜像。

## 能力边界

已验收主线：SIFT/DINO 对齐及差异复核、SAM 分割辅助证据、独立端口补漏、
有限可见线段到人工声明端点的拓扑复核、人工确认和复检记录。
端口增强默认关闭，保留人工复核；不自动确认电气连通性或真实故障。

已验收新增增强在重复使用的开发集上：TRAIN 295/344、inner 68/80、outer 40/56，
未匹配框分别 4/0/1；此前为289/344、65/80、33/56，旧命中无损失。
新原生候选训练头仅处理新增框，保留所有原算法结果，仍默认关闭；
分类器按源图分组验证，但上游检测器已见过 TRAIN，不是全检测器独立验证。
实际20来源、全部20便携后端精确复核及原项目实际Qt/Agent验收均完成；
已安装在 `inspection_agent/paired_native_pose.py`，沿用现有第四增强开关。
模型恢复工具也会从Release已校验的研究权重复制到正式output路径。
原E环境380项测试通过；公开快照379项通过，1项环境检查需先安装未打包的
独立SAM解释器。这不是算法失败，也不能宣称新机依赖安装已验收。
见 [夜间进展](PROGRESS_20261004.md)，不能把同场景开发指标当现场精度。

`research/experiments` 和 `research/evidence` 保留实验代码与阶段记录。
研究脚本中的 E 盘/工作目录路径来自原工作机，迁移时须配置；不能直接宣称全可移植。
原 E 盘项目的无关脏工作树及历史保持，只有已验收增强按请求接入；
此仓库是独立的公开备份快照，不是第二条算法主线。

## 第三方权利

SAM3 代码/模型适用 `licenses/SAM3-LICENSE`；DINOv2、LightGlue 保留原许可。
Ultralytics 模型及派生训练权重遵守上游 AGPL-3.0 或适用的单独授权。
本次没有替项目作者选择新的开源许可证，也没有消除第三方许可义务。
参阅各源码目录的许可证及 `licenses/THIRD_PARTY.md`。
