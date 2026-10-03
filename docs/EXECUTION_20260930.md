# ESDNet/UHDM 执行记录：2026-09-30

> 本页为 9 月 30 日至 10 月 1 日的历史执行记录。10 月 3 日发布验收的最终状态见 [发布结果](RELEASE_RESULTS.md)。

本记录接续 [核验](AUDIT_20260930.md) 和 [已批准计划](EXECUTION_PLAN.md)。
结果目录为 `results/release_20260930/`，暂存目录为 `outputs/release_20260930/`。

## 已验证

- `baseline.tar.gz` 保存开始修改前的 208 个源码/配置/证据文件；
  `baseline_manifest.json` 保存源码与 19 个模型包文件的哈希。
- `clean_build.json`：生产 CUDA 从空缓存成功编译，548.74 秒。
  CUTLASS commit `f7b19de32c5d1f3cedfc735c2849f12b537522ee`，CUDA 12.1，
  torch 2.2.2+cu121，RTX 4090。没有复用历史对象文件。
- `clean_build_pytest.xml`：原有 1,381 项测试通过，147.09 秒。
- `diagnostic_tests.xml`：新增 16 项诊断测试通过，覆盖两种位宽、四/八 warp、
  FewChannels、点卷积保留输入、两个消费者、特征省略、残差和 Graph 回放。
- `diagnostic_network_w4.json`：真实 4K 图像三个输出与显式无稀疏参考逐位一致，
  Graph 回放亦一致。157 层中 72 次后继量化、34 次第二消费者缓存、6 次残差融合、
  26 次特征省略，完整路径与无稀疏对照的调度次数完全相同。
- `diagnostic_network_w8.json`：同样三个输出和 Graph 逐位一致。完整/对照均为
  157 层、70 次后继量化、32 次第二消费者缓存、6 次残差融合、27 次特征省略。
- 完整测试集下载完成，1,000 个文件的大小和哈希已记录在数据目录 `manifest.json`。
  `data_manifest.json` 确认 200 对校准、24 对训练留出与 500 对测试之间无配对哈希重叠。
- `setup_kernel.log`：修改后的入口使用外部固定版本 CUTLASS 和 `uv sync --frozen`
  成功；38 个环境包校验完成。

## 最终调度下的新成本基线

真实图像 `0000_moire.jpg`，padding 后 2176×3840，每层两轮、每轮 30 个 CUDA
events 样本、Graph 内 3 次调用，顺序交替。完整网络由真实完整量化路径生成中间
激活。以下为各层中位数之和，不能视为整网时延。

| 位宽 | 完整路径 ms | 匹配主路径 ms | 整数核心 ms | R_core | R_pipeline | 单层 ≤30% |
|---|---:|---:|---:|---:|---:|---:|
| W4 | 59.664 | 45.185 | 24.478 | 59.15% | 32.04% | 28/157 |
| W8 | 74.160 | 60.327 | 36.201 | 38.21% | 22.93% | 34/157 |

证据：`branch_w4.json`、`branch_w8.json`。严格口径两者均未通过 30%。此前只统计
计算阶段的 W8 28.03% 不能代表包含稀疏准备后的预算。四 warp separate producer 的
主分支对照将仍需执行的输出打包融合进同一整数 tile，省略了稀疏结果中间往返。
核心计时为普通 FP16 输出（不含后继打包），包括特征省略 producer 的核心参考；
完整/主路径则保留实际特征省略。原始逐层样本与独立整网消融一并保存。

## 新入口

- `kernel/diagnostic.py` 从当前生产源代码生成独立的 dense 诊断扩展并编译，
  不依赖旧原型或 `.o`。适配锚点变化时显式报错。无稀疏的四 warp producer 保留
  整数 tile 和八元素输出访问，将后继打包放入 epilogue。
- `quantdemoire/branch_diagnostic.py` 提供显式上下文，只用于计时。模型包和环境
  变量不能开启此模式；离开上下文恢复原状态。该模式的输出不能用于质量报告。
- `script/verify_branch_diagnostic.py` 校验整网融合次数、dense 数值和 Graph 等价。
- `script/benchmark_branch.py` 用真实完整模型的中间激活测量全部量化层，分别
  记录完整路径、匹配的无稀疏路径、整数核心和独立整网消融。输出打包归 producer，
  剩余输入准备归 consumer。逐层中位数之和与整网时延分开报告。
- `script/download_uhdm.py` 提供带版本依赖的 uv script，校验完整 500 对镜像
  文件清单、字节数，支持复用/恢复，逐文件保存 SHA256。
- `script/setup_kernel.sh` 尊重外部 `CUTLASS_ROOT`，校验固定 commit，并允许
  `UV_BIN` 指定 uv 可执行文件；不再静默覆盖用户的 CUTLASS 路径。

## 数据来源和边界

[官方 UHDM 论文](https://arxiv.org/html/2207.09935) 明确 4,500 对训练图、500 对测试图。
[官方下载脚本](https://github.com/CVMI-Lab/UHDM/blob/main/scripts/download_data.sh)
给出 Google Drive 的 test.tar.gz 文件 ID `1bAZRuDq_bxZgxvnGtLwORV9pFKq2_kM_`。
此环境访问该公共 Drive 文件遇到 TLS 连接失败，因此使用原有数据相同的
[Kaggle 镜像](https://www.kaggle.com/datasets/soumikrakshit/uhdm-dataset)。
镜像中测试文件恰好为 `0000`—`0499` 的 1,000 张配对 JPEG，合计约 6.136 GB。
不能声称已核对官方压缩包的哈希；清单显式记录 `official_archive_hash_verified=false`。
独立完整数据目录是 `data/UHDM-full/test/`，原 100 对数据和旧结果保持可追溯。

## 论文基线与校准进展

完整 500 对 FP32 评估已完成，PSNR 22.1189937515、SSIM 0.7955921984、LPIPS
0.2550998695，四舍五入后与论文主表一致。证据 `quality_fp32_500_nchw.json`。
NHWC 严格 FP32 每图约 10.9 秒，cuDNN autotune 未改善；NCHW 约 0.38 秒，前三张
指标差异至多约 2e-6 dB，详见 `fp32_layout_comparison.json`。该布局变化只用于质量评估。

新增 `config/ptq/` 配方、稳定排序 FP16 尾部、dilation=2/4/8、ESDNet 的 VGG
预处理及循环余弦配置，具体假设见 [论文协议](PAPER_PROTOCOL.md)。`paper_protocol_tests`
有 9 项通过。200 对校准图从完整 4,500 对训练清单按 seed=123 抽取，已完成下载与哈希。
新校准子集和旧留出集有重叠，已重新选择不重叠的 24 对训练留出图；
`paper_data_manifest.json` 验证新校准/新留出/500 对测试均无配对哈希重叠。

W4 paper 配方首轮 4 epochs 校准完成，状态及冻结参数校验在
`outputs/ptq/esdnet_uhdm_paper_w4a4_s123/`，导出到 `checkpoints/ptq/paper-w4a4-v1/`。
`paper_w4_v1_format.json` 验证 157 层格式与尾部覆盖。留出集质量评估及后续修正见下表。
该轮发现池化后的 1×1 小 tensor 按 γ=0.001 只取一个样本，现有逐图均值统计可能
形成退化区间。用户已确认没有原始实现或状态，并授权适当提高 gamma。所有调整单独标记，
不把单值退化配置或调整后的配方称为严格逐参数复现。

## kernel 候选

**计时修正（2026-10-01）**：以下首轮候选使用的 Python 入口在每次调用中先构造
unpacked offsets，再构造 packed offsets，导致八 warp 路径的单项缓存反复失效。
这些额外 CUDA 操作进入了 Graph，因此旧屏测不能用于单独否定 kernel 本体。
入口已改为每次只请求正确格式，新增缓存对象复用断言；全部 340 项候选/诊断测试
通过。新的同批对照文件以 `_w4_cached.json` 结尾。旧文件不删除、不覆盖。
生产路径和主分支诊断没有这次重复构造问题，之前的严格分支预算报告仍有效。

直接从 packed NHWC 读取稀疏激活的候选通过 60 项算术、边界、残差、prefix 和 Graph
检查，但性能退化：W4 20 个热点合计 29.578 → 56.874 ms，W8 6 个热点
14.421 → 36.947 ms。输入为合成张量、形状和权重来自实际部署，尚不含输出消费者
缓存，因此只是筛选测量，不能当作 E2E。该候选不采用；源码/结果归档在
`outputs/release_20260930/spatial_packed_rejected.tar.gz`。

保持 planar 合并访问、一次提前加载两个 CSR 项的 FP32 累加版本通过 60 项测试，
但 W4 20 个热点均未获益，合计 29.358 → 31.464 ms（0.9331×）。不进入默认路径。
继续测试预计算激活码与 FP16 权重乘积的 lookup 候选，保留 FP32 加法次序。

新增 `script/reproduce_ptq.py`，串起原始 checkpoint 校准、mask 不变的导出、格式验证
及 fake/CUDA 评估。小 tensor 单值采样保护作为显式独立选项，W4 对照训练已完成。
新增按 manifest 验证的留出集评估入口，含路径、内容哈希和重复项检查（2 项测试通过）；
论文配置、loss 与采样规则检查共 10 项通过。

## 2026-10-01 后续实测与决定

修复候选入口缓存后重新屏测 W4 的 20 个热点（两个交替轮次，合成输入、实际权重，
不含后继缓存；不是整网测量）：

| 候选 | 同批生产 ms | 候选 ms | 加速比 |
|---|---:|---:|---:|
| packed NHWC 直接读取 | 29.774 | 56.361 | 0.528× |
| 两项预取 | 29.645 | 31.185 | 0.951× |
| 精确乘积 LUT | 29.770 | 33.146 | 0.898× |
| FP16 half2 累加 | 29.622 | 30.810 | 0.961× |
| planar INT4 压缩 | 29.376 | 29.620 | 0.992× |
| 16 lane CSR 行 | 29.621 | 30.976 | 0.956× |

全部候选均未纳入生产代码。planar INT4 的八 warp 子集仅约 1% 收益，四 warp 回退，
不足以承担全网 producer 缓存格式迁移。half2 改变累加精度，仅通过独立半精度参考，
不沿用逐位相等结论，也没有图像质量通过结论。报告索引 `spatial_candidates_w4_cached.json`。
实验测试默认跳过，可用 `pytest --run-experimental` 运行；340 项实验与诊断检查通过。
静态 kernel attributes 保存于 `experimental_resources.json`，不等同于实际 profiler 归因。

用户允许提高两个 gamma。固定 200 张校准图、4 epochs 和训练参数，所有配方仅使用
24 对互斥训练留出图的 512 中心裁剪选择，指标为 PSNR / SSIM / LPIPS：

| 配方 | W4 | W8 |
|---|---|---|
| FP32 参考 | 22.3031 / .81150 / .19284 | 同左 |
| γ=.001，单值保护 | 20.0317 / .71408 / .35466 | 21.5248 / .80274 / .19776 |
| γ=.01，单值保护 | 20.2827 / .76554 / .26874 | 20.9609 / .79647 / .20786 |
| γ=.01，池化层完整统计 | 21.1676 / .77476 / .23955 | 21.8294 / .80665 / .19558 |
| γ=.05，池化层完整统计 | 20.8362 / .77112 / .24689 | 21.9682 / .80799 / .19402 |

原始单值退化 W4 为 16.7486 dB；跨图 extrema 对照为 8.6926 dB，均弃用。
γ=.01 时部分池化 tensor 从 1 个样本变为 2–3 个样本，绕开原来的单值保护，构成混杂。
新增 `full_if_pooled` 显式规则：NCHW 的 H=W=1 使用完整 tensor 范围，其余仍按 gamma
采样并保留单样本保护；先采样再回退，固定同一 gamma 下的 RNG 消耗。
W4 γ=.01 单值保护的 FP32 fake oracle 为 20.3092 dB，与对应 CUDA 差 .0265 dB。
所有新导出均通过逐项 quant_state 身份核对，mask、FP16 尾部、整数码和量化参数不变。
结果索引 `ptq_holdout_comparison.json`，这些数字不能当作完整测试集论文主表。
γ=.05 同规则对照已完成；按留出 PSNR 选择 W4 γ=.01、W8 γ=.05。
测试前冻结记录为 `paper_adjusted_selection.json`，包括候选和最终模型哈希。

部署结构分析 `smoothing_structure.json`：在保持两侧各 .25%、真实 nnz=29,720 不变时，
用正常 SmoothQuant alpha 从 .5 调到 1，估计空间卷积每次前向累计 planar 写入量
从约 2.478 GB 降到 1.223 GB。这是跨层累计字节数，既不是峰值显存，也不是实测速度。
需重新校准并验证质量；论文配方 alpha 仍固定 .5。

## 待继续

- 冻结训练留出选择后的 W4/W8 调整 PTQ 配方，完成 500 对 fake/CUDA 质量评估。
- 校准部署 .25% 尾部和 alpha 对照，实测分支预算及完整端到端性能。
- 30% 严格目标两种位宽均未达标，保持为待完成项。
- 完成最终同权重、同 profile 的 FP32/FP16 显存对照及 HF 本地格式包。

## 后续部署与环境验证

原始 checkpoint PTQ 的部署 .25% 尾部、W4 γ=.01、alpha=.5 已完成：训练留出
21.35847 / .776884 / .250908。结构与导出逐项一致。移植同架构调度仅作为
`status=unmeasured_proposal`，不移植旧权重的速度/质量结论；浮点边界不使用
整数主输出的格式标记，兼容检查仅在整数层核对 main_output。
全部 157 层成本重新测量：mixed 59.54314 ms，main_only 45.23718 ms，core
24.41346 ms，R_core 58.5986%，R_pipeline 31.6243%，30/157 层 ≤30%。
证据 `branch_deploy-w4a4-full_if_pooled-g0.01.json`，仍未通过严格目标。

第七个候选 `sparse_tc_coalesced` 先统计 8 行坐标 union：主要热点在补零后的
FP16 项数约为真实 nnz 的 7.1–9.0 倍。采用合并空间加载及共享内存组织 MMA，
独立 FP64 稀疏参考和变化输入 Graph 共 12 项通过，但 W4 20 热点为
29.39817 → 34.14604 ms（0.861×），不采用。改变累加次序，没有发布其图像分数。

独立源码目录开始执行 `uv sync --frozen`；离线尝试因缓存缺 wheel 而失败，
随后联网从锁文件下载，未修改锁定版本。安装与后续检查的日志单独保存。

W4 alpha=1 的实际结果：训练留出 11.16586 / .465603 / .638025；完整图 Graph
约 75.43–75.64 ms，但 R_core 49.7882% 仍不通过（mixed 57.42973、main_only
45.27153、core 24.41986 ms）。质量严重退化，明确弃用，不能仅凭 staging 字节
减少或约 2 ms 整网收益选择该配置。

干净目录 `uv sync --frozen` 已安装 38 个锁定包，`uv pip check` 全部兼容；
导入路径指向独立目录自身，22 项 CPU 协议/数据/profile 检查通过。
证据 `clean_install_uv_online.log`、`clean_install_imports.json`；独立测试日志/XML
位于 `outputs/release_20260930/clean_install_cpu_tests.*`。导出新增原始 checkpoint、
校准配置校验与配方匹配，防止错误混用浮点边界权重或标错 gamma；对应独立测试通过。

## 完整测试评估结束与发布收尾

后台完整测试评估已全部成功结束。500 对 FP16 为 22.118892 / .795589 / .255058；
两侧各 .5% 的 adjusted PTQ：W4 fake/CUDA 为 20.971699 / 20.973432 dB，
W8 fake/CUDA 为 21.868575 / 21.868181 dB。完整 SSIM、LPIPS 和逐图结果见
`quality_paper_adjusted_w{4,8}_{fake,cuda}_500.json`。这些结果与论文 W4 21.08、
W8 22.00 dB 仍有差距，不能称作精确复现。

生产回归 1,415 项通过，360 项研究测试默认跳过。并发 CSR 分支独立 24 项检查
通过，但 W4 20 个热点合计 29.54075 → 41.94057 ms（0.704×），不采用。
W8 alpha=1 留出 PSNR 20.80086，R_core 32.8781%，质量退化且预算仍失败。
.25% 部署候选按留出质量冻结为 alpha=.5：W4 gamma=.01、W8 gamma=.05。

用户现要求准备 push 最终代码版本，收尾执行范围见 [PUSH_PLAN.md](PUSH_PLAN.md)。
尚未提交新增代码或执行远程 push；最终 .25% 部署包的完整评估与最终 profile 验收
列为发布整理中的必做项。
