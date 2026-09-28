# 文件索引

本仓库共 **168 个文件**，全部放在根目录（网页/远端不分子目录，仅 `figs/` 与 `web/` 两个目录）。
下表按用途分成 10 类，便于快速定位。

> 安全约定：`atlas.env`（设备口令）、`.ssh/`、`.github_token`、`*.pem` / `*.key` 一律在 `.gitignore` 里，
> 不会进入仓库；仓库内只有 `atlas.env.example` 占位模板。模型的 `.om` / `.onnx` / `.safetensors` 体积大，
> 只留在设备上（`/home/disk/models/`），也不入库。

---

## 1. 报告与文档（12）

| 文件 | 内容 |
| --- | --- |
| `README.md` | 总览：关键结果、环境、仓库结构、部署与访问、复现顺序、踩坑记录 |
| `PROGRESS.md` | 逐日进度日志，最新在最上面 |
| `报告一_补实验协议前（2026-09-06~09-20）.md` | 汇报材料：协议前的部署与可用化，末尾给出交给补实验的三个问题 |
| `报告二_补实验协议后（2026-09-20~09-28）.md` | 汇报材料：协议后 P0–P6 全记录，含口径约定、负面结论、工程踩坑与局限 |
| `02_补实验执行协议.md` | 合作方给定的补实验协议（P0–P6），本仓库实验按它执行 |
| `Atlas500A2_NPU最终验证报告_2026-09-07.md` | 阶段性主报告（21 节） |
| `P0_校准与参考一致性_报告.md` | P0：校准时序错误定位 + 重放 / 双缓冲 / 参考链路一致性 |
| `P2_量化对照实验_报告.md` | P2：int8 对照（§9 P2-C 注入、§10 P2-D 脉冲、§11 P2-F 强制、§12 P2-B 等样本数） |
| `P3_混合精度_报告.md` | P3：逐层敏感度、帕累托、H 窗口截断（§8）、方向指标（§9） |
| `P4_基线评估_报告.md` | P4：lm-eval 接入、PIQA/LAMBADA 子集、CPU↔OM 逐 token 对照、成本、跨规模迁移 |
| `P5_性能矩阵_报告.md` | P5：4 臂 × 5 前缀 × 5 重复矩阵、HTTP 真实延迟、内存分解 |
| `P6_图表与消融_报告.md` | P6：六幅论文用图与关键消融表（含 G=I 消融） |

## 2. 图与出图数据（9）

| 文件 | 内容 |
| --- | --- |
| `figs/fig1_pareto.png` | 图 1 质量—延迟 Pareto |
| `figs/fig2_trajectory.png` | 图 2 按 token 步的误差轨迹 |
| `figs/fig3_metric_vs_cost.png` | 图 3 候选指标：排序质量 vs 测量成本 |
| `figs/fig4_perf_memory.png` | 图 4 性能与内存分解 |
| `figs/fig5_ablations.png` | 图 5 关键消融 |
| `figs/fig6_search_cost.png` | 图 6 搜索成本 |
| `p6_figures.py` | 出图脚本（读设备侧原始 JSON，可复现六幅图） |
| `p6_metrics.json` | 图用指标面板 |
| `p6_ablations.json` | 消融表原始数据 |

## 3. 推理引擎与服务（20）

| 文件 | 说明 |
| --- | --- |
| `rwkv7_serve2.py` | pyACL 零拷贝推理引擎（逐层绑定、状态原地更新、档位加载） |
| `rwkv7_chat.py` | 设备端 CLI 对话（system 提示词、状态缓存、停止符、重复惩罚、流式输出） |
| `rwkv7_http.py` | 常驻 HTTP 服务：`/chat`、`/chat/stream`(SSE)、`/health`、`/tier` + 静态网页托管 |
| `web/index.html` | 网页前端（单文件、无框架、无 CDN）：气泡对话、代码块复制、长度与档位选择 |
| `device_rwkv7_chat.py` / `device_rwkv7_serve2.py` | 设备侧包装层（stdin 编码兜底、档位与模型目录解析） |
| `rwkv7_serve.py` | 早期服务版本（保留用于对照） |
| `rwkv7_acl_run.py` | pyACL 直接跑 `.om` 的最小可运行示例 |
| `atlas500a2_acl_demo.py` | 最早的 ACL 连通性 demo（设备就绪性检查） |
| `rwkv7_cpu_ref.py` / `rwkv7_step_ref.py` | CPU 参考实现（aarch64 bf16 需关 MKLDNN） |
| `rwkv7_cmp_om.py` | `.om` 与 CPU 参考的逐层数值对比 |
| `rwkv7_profile.py` | 逐层耗时/内存剖析 |
| `rwkv7_demo.py` | 最早的单文件推理 demo |
| `modeling_rwkv7.py` / `configuration_rwkv7.py` | RWKV-7 模型与配置（HuggingFace 风格，用于导出与参考） |
| `tiers.json` | 三档位表（精确 fp16 / 均衡 top8 / 快速 top16） |
| `ask.sh` / `ask_q.sh` | 命令行提问小工具（带长度参数） |
| `ask_http.py` | 走 HTTP 接口提问的小工具 |

## 4. 量化与 ATC 编译（28）

链路：采集校准数据 → 逐层量化 → 选层 → ATC 编译 → 清单与哈希。

| 文件 | 说明 |
| --- | --- |
| `capture_calib.py` / `capture_calib_pre.py` / `capture_calib_p1.py` | 校准数据采集（`pre` = 修正时序前，`p1` = 扩展数据：64 段语料 × 状态年龄 0/64/256） |
| `make_head_calib.py` | 输出头（head）校准 |
| `quantize_layer.py` / `quantize_layer_calib.py` / `quantize_layer_sel.py` | 单层 int8 量化（按目标层选择性量化） |
| `rwkv7_build_layers.sh` / `build_int8_layers.sh` / `build_calib_layers.sh` / `build_cp_layers.sh` / `build_ffn_layers.sh` / `build_p1_layers.sh` | 逐层 ATC 编译链（单 worker 串行，规避 60 秒硬件看门狗） |
| `rwkv7_export_head.py` / `rwkv7_export_chunk.py` | 导出 head / 分块 ONNX |
| `rwkv_onnx_npu.py` / `rwkv_onnx_fix_dequant.py` | ONNX 图改写（NPU 算子适配、反量化修正） |
| `make_manifest.py` / `update_manifest_p1.py` | `.om` 清单（sha256）生成与回填 |
| `make_plans_p1.py` / `plan_both.py` / `pack_calib_p1.py` / `p1_corpus.py` | 选层方案生成、校准语料打包 |
| `plan_15b_m_top8.json` | 1.5B 的 top8 选层方案 |
| `quant_test.py` | 量化前后数值冒烟测试 |
| `sweep_int8.sh` / `sweep_int8_device.log` | 逐层 int8 扫描脚本与设备日志 |
| `toggle_int8.sh` | int8 档位切换辅助 |

## 5. 实验脚本（61）

P0–P5 与通用分析脚本；P6 的出图脚本与数据见第 2 节。

**P0 时序（2）**
`p0b_replay.py`（重放）、`p0c_alias.py`（双缓冲 alias 验证）

**P1 校准数据（5）**
`p1_chain.sh`（全链）、`p1b_chain.sh`（续跑）、`pilot_recalib.sh`（小规模预热）、`probe_cq.sh` + `probe_cq_device.log`（设备侧探针）

**P2 错误来源分解（9）**
`p2b_chain.sh`（等样本数对照）、`p2c_inject.py` + `p2c_ref_states.py`（注入 vs 自由）、`p2d_pulse.py`（脉冲 vs 持续）、`p2df_chain.sh`、`p2f_free_gen.py` + `p2f_only_chain.sh`（强制重放对照）、`cmp_layer_numeric.py` / `cmp_layer_real.py`（数值一致性）

**P3 混合精度与指标（16）**
`p3_chain.sh`、`sensitivity.py` + `sensitivity_p1.json`（逐层敏感度）、`pareto.py`（帕累托）、`p3_combine.py`（合并结果）、`p3_local_chain.sh` + `p3_local_metrics.py` + `p3_layer_err.py`（局部指标）、`p3_horizon.py`（H 窗口截断）、`p3b_dir_chain.sh` + `p3b_err_vectors.py` + `p3b_dir_metric.py` + `p3b_compare.py`（方向指标）、`probe_layer_err.sh` + `probe_layer_err_device.log`、`diag_trim.py`

**P3c 跨规模迁移（6）**
`p3c_15b_quant_chain.sh`、`p3c_15b_resume_chain.sh`、`p3c_15b_finalize_chain.sh`、`p3c_compare_depths.py`、`p3c_15b_compare.json`、`sensitivity_15b.json`

**P4 基线评估（14）**
`p4_score_server.py` + `p4_lm_eval_backend.py` + `p4_run_eval.py`（lm-eval 自定义 backend）、`p4_pilot_chain.sh`（子集评测）、`p4_dump_items.py` / `p4_compare_items.py` / `p4_score_items.py`、`p4_cpu_check_chain.sh` + `p4_token_check.py` + `p4_token_check_chain.sh` + `p4_token_npu_rerun.sh`（CPU↔OM 逐 token 对照）、`p4_smoke_test.py`、`p4b_15b_chain.sh` + `p4b_download_15b.py`（1.5B 部署）

**P5 性能（5）**
`p5_bench.py`（固定工作量矩阵）、`p5_chain.sh`、`p5_http.py`（HTTP 真实延迟）、`p5b_chain.sh`、`bench_ab.sh`

**通用分析（4）**
`make_summary.py`（汇总表生成）、`ab_compare.py` + `ab_run_arm.py` + `ab_logits_device.json`（A/B 逐 token 对照）

## 6. 运维与电源（12）

| 文件 | 说明 |
| --- | --- |
| `start_service.sh` | 启动/停止常驻服务（`stop` / 8000） |
| `service_supervisor.sh` | 服务监督（退出码 75 → 按新档位重启） |
| `check_service.sh` | 每 5 分钟健康检查（crontab 注册） |
| `boot_all.sh` | 开机自启：等磁盘挂载 → 拉起 `slogd`/`dmp_daemon` → 起服务 |
| `mem_guard.sh` | 内存守卫（可用内存过低时收敛） |
| `experiment_guard.sh` | 实验锁：实验期间暂停看门狗，避免抢内存 |
| `quality_cmp.sh` | 质量巡检对比 |
| `auto_poweroff.sh` / `auto_poweroff_cp.sh` / `auto_poweroff_sens.sh` / `auto_poweroff_wrap.sh` | 跑完自动关机（等进程结束 → 落现场 → 停看门狗 → 释放锁 → sync → `poweroff -f` 三重试） |
| `p5_wrapup_poweroff.sh` | P5 收尾脚本（汇总 + 关机） |

## 7. 测试（4）

| 文件 | 说明 |
| --- | --- |
| `test_web.py` | 网页版验收（静态托管、流式、多轮、路径穿越防护） |
| `test_chat_stream.py` | SSE 流式接口测试 |
| `test_tier.py` | 档位切换测试 |
| `test_trim.py` | 停止符/复读截断测试 |

## 8. 工具（4）

| 文件 | 说明 |
| --- | --- |
| `atlas.py` | 远程执行助手：密码登录 + `develop` 提权 + SFTP 上传下载，凭据从 `atlas.env` 读 |
| `atlas.env.example` | 凭据模板（复制成 `atlas.env` 后填真实值；`atlas.env` 不入库） |
| `push_repo.ps1` | 无交互推送到 GitHub（令牌从环境变量 / `.github_token` / 凭据库按序取，不落进 URL） |
| `.gitignore` | 凭据、模型与构建产物、设备运行期文件的忽略规则 |

## 9. 参考资料（2）

| 文件 | 说明 |
| --- | --- |
| `ascend_pytorch_700.pdf` | 昇腾 PyTorch 框架适配文档（离线留存） |
| `ascend_pytorch_700.txt` | 上者的文本提取版，便于仓库内检索 |

## 10. 协议收尾补充（2026-09-27 ~ 09-28 新增，15 个）

补齐协议里此前缺的项：H=128 窗口、跨分布复核、权重 MSE、逐题对照、ARC/HellaSwag 子集、
稠密 Gramian 一致性。

| 文件 | 说明 |
| --- | --- |
| `p3d_horizon.py` | P3-D：128 步窗口 + 跨分布（pilot/p1test）的排序相关、top-k 重合、配对 bootstrap |
| `p3d_chain.sh` / `queue_next.sh` / `queue_tail.sh` | P3-D 链与两个串行队列（一次只跑一个重活） |
| `p3d_probe.py` | 采臂前的数据口径探测（既有臂步数、各文本 token 数） |
| `p3w_weight_mse.py` | P3-W：权重 MSE（先在一份存量量化图上标定口径，再算 32 层 + 输出头） |
| `p3w_compare.py` | 权重 MSE 与 rollout KL 的排序相关性（结论：无预测力） |
| `p3w_probe.py` | 量化 ONNX 结构探测（int8 权重如何存、AscendQuant/Dequant 语义） |
| `p4_item_cpu.sh` | P4 ① 逐题 CPU↔NPU 对照的 CPU 侧补跑 + 比对 |
| `p4_arc_hella_chain.sh` | P4 ⑤ ARC-Easy / HellaSwag 子集（fp16 与均衡档） |
| `p3e_gramian.py` / `p3e_chain.sh` | P3-E：稠密 Gramian 与结构化递推的一致性 + 离线耗时 |
| `queue_after_arc.sh` / `finish_queue.sh` | 等前序链结束再补跑（严格串行；含 9-28 并发事故后的防复发逻辑） |
| `final_poweroff.sh` | 收尾关机守护（**已停用**：这台设备关机流程连续两晚卡死，改人工断电） |

另外 `ab_run_arm.py` 加了分阶段计时与 `take_ids()` 前缀分词（后者把单臂 336 s 压到 85 s），
`p4_run_eval.py` 加了 `jsonable()` 防止结果 JSON 被函数对象截断。
