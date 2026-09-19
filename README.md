# 基于昇腾 Atlas 500 A2 的 RWKV 课程助手

在华为 Atlas 500 A2 智能小站（昇腾 310B NPU，ARM64）上**本地部署** RWKV-7 G1 2.9B，
做成可网页访问的课程问答助手。模型全部在本机 NPU 上推理，不依赖外网。

## 关键结果

| 指标 | 数值 |
| --- | --- |
| 生成速度 | 5.5 ~ 5.75 token/s（约 180 ms/token） |
| 预填充 | 3 ~ 5 秒（系统提示词状态缓存后，只跑用户那一句） |
| 首字延迟 | 4 ~ 7 秒（SSE 流式输出） |
| 模型加载 | 约 43 秒（常驻服务只加载一次） |
| 权重体积 | 5.2 GB（fp16），运行时内存约 5.5 GB |
| 设备上限 | 2.9B；7.2B 受 11.5GB 共享内存限制，本机不可行（见下） |

## 技术路线

```
PyTorch(safetensors) → 逐层导出 ONNX → ATC 编译成 .om → pyACL 零拷贝串层推理 → HTTP/网页
```

每个 Transformer 层编译成一个独立的 `.om`，层间张量复用同一块设备内存原地更新（零拷贝），
每 token 只有 1 次 H2D（embedding 10KB）和 1 次 D2H（logits 262KB）。

## 环境

- 硬件：Atlas 500 A2 智能小站，昇腾 310B，4 核 ARM64，**11.5 GB 共享内存**（无独立显存）
- 系统：EulerOS + CANN 8.0.RC1（NNRT）+ pyACL
- 模型：RWKV-7 G1 2.9B（32 层，hidden 2560，40 heads，vocab 65536）
- Python：Miniconda 环境 `npu22`（推理）、`rwkv7`（导出）、`quant`（量化）

## 仓库结构

**推理与服务（设备侧）**

| 文件 | 说明 |
| --- | --- |
| `rwkv7_serve2.py` | pyACL 零拷贝推理引擎（逐层绑定、状态原地更新） |
| `rwkv7_chat.py` | 对话层：system 提示词、状态缓存、停止符、重复惩罚、流式输出 |
| `rwkv7_http.py` | 常驻 HTTP 服务：`/chat`、`/chat/stream`(SSE)、`/health` + 静态网页托管 |
| `web/index.html` | 网页前端（单文件，无框架/无 CDN）：气泡对话、代码块复制、长度选择 |
| `ask.sh` / `ask_q.sh` / `ask_http.py` | 命令行入口（服务在跑时自动走 HTTP，避免重复加载模型） |
| `start_service.sh` / `boot_all.sh` / `check_service.sh` | 起停、开机自恢复、健康看门狗 |

**模型转换与编译**

| 文件 | 说明 |
| --- | --- |
| `rwkv7_export_chunk.py` | 逐层导出单步 ONNX（权重按层从 safetensors 读取，避免整模入内存） |
| `rwkv7_export_head.py` | 导出输出头（65536×2560）ONNX |
| `rwkv7_build_layers.sh` | 逐层 ATC 编译脚本 |

**量化实验（结论见下节）**

| 文件 | 说明 |
| --- | --- |
| `capture_calib.py` / `make_head_calib.py` | 抓取真实激活值作为量化校准数据 |
| `quantize_layer.py` | label-free int8 量化 |
| `quantize_layer_calib.py` | 带真实数据校准的 int8 量化 |
| `quantize_layer_sel.py` | 自研选择性量化（白名单指定只压哪些矩阵） |
| `build_int8_layers.sh` / `build_calib_layers.sh` / `build_ffn_layers.sh` | 三套量化流水线 |
| `cmp_layer_numeric.py` / `cmp_layer_real.py` | 逐层误差对比（随机输入 / 真实输入） |
| `probe_layer_err.sh` / `sweep_int8.sh` / `probe_cq.sh` / `toggle_int8.sh` | 误差扫描与质量抽检 |
| `bench_ab.sh` | fp16 / int8 对照测试 |
| `mem_guard.sh` / `auto_poweroff.sh` | 内存看门狗、跑完自动关机 |

**工具与文档**

| 文件 | 说明 |
| --- | --- |
| `atlas.py` | 远程执行助手（密码登录 + `develop` 提权 + SFTP），凭据从 `atlas.env` 读 |
| `test_web.py` | 网页版验收脚本（静态托管、流式、多轮、路径穿越防护） |
| `PROGRESS.md` | 逐日进度与结论记录 |
| `Atlas500A2_NPU最终验证报告_2026-09-07.md` | 阶段性验证报告 |
| `*_device.log` | 设备上跑出来的实验数据（误差扫描、层数扫描） |
| `ascend_pytorch_700.*` | 华为官方文档参考资料 |
| 其余 `rwkv7_*.py` / `rwkv_onnx_*.py` | 早期探索版本，保留备查 |

## 怎么用

### 1. 部署（设备上）

```bash
# 环境
source /home/disk/cann80base/ascend-toolkit/set_env.sh
# 逐层导出 + 编译（已有 .om 会自动跳过）
bash rwkv7_build_layers.sh 0 31
# 起常驻服务
bash start_service.sh 8000
```

设备重启后 `slogd` / `dmp_daemon` 不会自启（不拉起则 `npu-smi` 报 -8010、NPU 不可用），
`boot_all.sh` 已通过 crontab `@reboot` 自动处理；`check_service.sh` 每 5 分钟自检。

### 2. 访问

- 网页：`http://<设备IP>:8000/`
- 命令行：`./ask.sh "什么是熵？" 128`
- 接口：`POST /chat`、`GET /chat/stream?q=...&tokens=128`（SSE）、`GET /health`

### 3. 远程操作（开发机）

```bash
cp atlas.env.example atlas.env   # 填入设备地址与密码（不会提交）
python atlas.py run "npu-smi info"
python atlas.py put 本地文件 /home/disk/...
```

## 量化实验结果（重要）

目标是进一步提速，但在这台设备上**int8 无法兼顾质量**，实测数据：

| 方案 | 速度 | 体积 | 输出质量 |
| --- | --- | --- | --- |
| fp16（最终采用） | 5.3 ~ 5.75 tok/s | 163 MB/层 | 正常 |
| label-free int8 全量 | 11.6 tok/s（2.2×） | 86 MB/层 | 崩坏（复读/乱码） |
| 校准 int8 全量 | 7.9 tok/s | 90 MB/层 | 8 层以上退化 |
| 只压 FFN 的自研方案 | 6.4 tok/s | 106 MB/层 | 8 层开始退化 |
| 仅输出头 int8 | 6.2 tok/s | 169 MB | 崩坏 |

**根因**：各层激活幅度相差近百倍（x 峰值 0.05 ~ 8.2，wkv 状态峰值 1 ~ 146），而工具链用的是
离线静态量化刻度，长尾把有效分辨率吃掉了——实测每层循环状态 wkv 的相对误差达 20%~55%，
32 层递归放大后模型失忆。用真实数据校准能把单层误差压到 1.7%~9.5%（wkv 误差降低约 12 倍），
但仍然撑不过 8 层。

**为什么不能做得更好**：CANN 8.0.RC1 的 ONNX 量化工具缺少 LLM 量化的关键能力——没有
per-channel/per-token 激活量化、没有 SmoothQuant、也没有仅权重量化（W8A16）。设备带宽大头
是权重（每 token 读 5.2 GB），激活只占 22 MB，所以"只压权重、激活保 fp16"这条路本可以对症，
但工具链里没有对应的可用算子。

## 7.2B 可行性

| 配置 | 权重占用 | 结论 |
| --- | --- | --- |
| 7.2B fp16 | 约 14.4 GB | 超过整机 11.5 GB，装不下 |
| 7.2B int8 | 约 7.2 GB | 加系统与运行时约 11.2 GB，贴死上限且无 swap，工程上不可行 |
| 7.2B int4 | 约 3.6 GB | 能装，但精度会比已不可用的 int8 更差，且 310B 的 int4 矩阵乘未验证 |

结论：**本机合理上限是 2.9B**，7.2B 需要 32 GB 级显存设备，或采用"边缘端文本 + 云端大模型"的协同方案。

## 踩坑记录

1. **容器方案不可用**：官方镜像编译时绑死了 HCCL 的 `libhccl.so`（单卡推理并不需要它），
   挂载宿主目录与安装 NNRT 都绕不过去，最终改为宿主机直跑。
2. **重启后 NPU 不可用**：`slogd` 与 `dmp_daemon` 不自启，`npu-smi` 会报 `-8010`，必须手动拉起。
3. **60 秒硬件看门狗**：设备 `wdTimeout=60s, wdAction=2`；ATC 多 worker 并行编译会把
   4 核 11GB 的机器压到假死甚至复位，**只能单 worker 串行**（约 6 分钟/层）。
4. **一份模型占 5.5 GB**：同时跑两个推理实例会把机器拖死，`ask.sh` 因此改为优先走 HTTP。
5. **pyACL context 绑定线程**：多线程 HTTP 服务里执行推理会报 `107002`，服务必须是单线程。
6. **HTTP/1.1 keep-alive 会堵死单线程服务**：浏览器空闲长连接让服务器阻塞在读取上，
   新请求排队超时；改用 HTTP/1.0（一条连接一个请求）后解决。

## 安全说明

仓库内**不包含**设备密码与 SSH 私钥：`.ssh/`、`atlas.env`、`*.pem` 均已加入 `.gitignore`；
`atlas.py` 的凭据从环境变量或本地 `atlas.env` 读取。

## 后续可做

- 多轮会话隔离（引擎已支持状态快照导入导出，每份 22 MB，切换约 0.1 秒）
- 网页访问口令、对话历史持久化
- 批量 prefill（模型代码使用 `aten::linalg_solve_triangular`，ONNX 导不出，需改写该段数学）
- 图文问答：图片交给外部视觉 API 转文字后拼入 prompt，本机侧无需改动
