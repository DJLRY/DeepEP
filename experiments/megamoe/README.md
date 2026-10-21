<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# MegaMoE Prefill / Decode Beta

公开入口：`deep_ep_experimental.megamoe`。W4A8 共享专家按实际单卡 B 自动选择小 BS / Prefill 策略；路由专家保留 Prefill 调度。
构建和多机测试入口位于本目录的 `scripts/`；机器地址及环境配置由使用者提供。

## 1. 输入、输出与公开接口

定义：`B` 为每卡 token 数，`H` 为隐藏维度，`I` 为 SwiGLU 后宽度，`K` 为 top-k，
`E` 为全局路由专家数，`L=E/EP` 为每卡路由专家数，`S` 为每卡共享专家数。

```python
# 公开调用签名；原位写入 y，返回 None。
def fp8_fp4_mega_moe(
    y, l1_weights, l2_weights, sym_buffer,
    shared_l1_weights=None, shared_l2_weights=None,
    cumulative_local_expert_recv_stats=None,
    recipe=(1, 1, 32), activation="swiglu",
    activation_clamp=None, fast_math=True,
    *, x=None, timer=None,
) -> None: ...
```

| 参数/输入 | 类型、形状与用途 |
| --- | --- |
| `y` | 调用方预分配的 BF16 `[B,H]` 输出，不能与输入重叠 |
| `sym_buffer.x / x_sf` | 默认输入：FP8 E4M3 `[B,H]` + E8M0 `[B,H/32]`，调用前填入 buffer 的前 B 行 |
| `x` | 可选 BF16 `[B,H]`，覆盖 buffer 中的 FP8 输入；内部量化为 FP8，**不是 BF16 GEMM** |
| `sym_buffer.topk_idx` | int32 `[B,K]`，全局专家 ID，同 token 不重复且不能为 -1 |
| `sym_buffer.topk_weights` | FP32 `[B,K]` 路由权重；BF16/FP8 输入均从 buffer 读取路由 |
| `l1_weights / l2_weights` | 两对 `(weight, scale)`，当前测试使用 FP4 E2M1 权重 + E8M0 scale |
| `shared_l1_weights / shared_l2_weights` | 共享专家的两对权重，必须同时传入，数量与 buffer 的 S 一致；默认不使用 |
| `cumulative_local_expert_recv_stats` | 可选 int32 `[L]`，在原值上累加专家接收 token 数 |
| `timer` | 可选 `buffer.allocate_timer()`；仅在单独诊断调用中传入 |

`recipe/activation/activation_clamp/fast_math` 当前仅支持签名中的默认值。所有张量在 buffer 所在 NPU 上。
FP8 输入仍需打包到通信区，因此流水中的 `AIV_QUANT` 此时表示打包，不是重新量化。

权重在测试/推理前转换一次，转换过程不计入性能测量：

| 原始 ND 权重 | packed FP4（uint8，两个值/字节） | E8M0 scale |
| --- | --- | --- |
| GMM1 | `[L,2I,H/2]`，N 方向先 gate 后 up | `[L,2I,H/64,2]` |
| GMM2 | `[L,H,I/2]` | `[L,H,I/64,2]` |

调用 `w1, w2 = transform_weights_for_mega_moe(raw_w1, raw_w2)` 得到 Ascend NZ 权重。FP4 routed GMM1 一次准备两套 weight/scale：实际 B≤256 使用 `[gate128, up128]` 融合布局，B>256 使用 `[gate | up]` 配对布局。转换不进入计时区间，但额外保留一份 GMM1 权重和 scale；GMM2 不复制。
请完整保留返回的 `w1`，不要解包后重组、原地修改或自行搬运其张量；权重更新或设备变化时从原始 ND 重新转换。共享专家将首维 L 换成 S，并传入 `shared=True`，使用 planar 权重布局。
不能直接传入已经做过 CUDA 专用重排的权重。

buffer 的构造参数如下，模型参数含义与测试 JSON 相同：

```python
from deep_ep_experimental.megamoe import (
    StoreGroup, SymmBuffer, fp8_fp4_mega_moe, transform_weights_for_mega_moe,
)

# torchrun 启动后先 torch.npu.set_device(local_rank)。
group = StoreGroup.from_env()  # 主机协调用 TCPStore，算子通信用 SHMEM
buffer = SymmBuffer(
    group, num_experts=384, num_max_tokens_per_rank=4096, num_topk=6,
    hidden=7168, intermediate_hidden=3072,
    num_shared_experts=0, mma_type="fp8xfp4", activation="swiglu",
    ranks_per_node=8, timeout=120,
)
```

准备输入后依次调用 `buffer.validate_inputs(B)`（BF16 时额外传入 `x=x_bf16`）、`buffer.prepare(B)` 和算子。
执行成功后，所有 rank 按相同顺序调用 `buffer.destroy()`、`group.close()`；
异常处理及完整可运行调用示例见 [example.py](python/deep_ep_experimental/megamoe/example.py)。

约束：EP=2～128，E 可整除 EP 且每卡最多 128 个路由专家；S=0～8；B=1～65536；
H=1024～8192 且为64的倍数，I=256～4096 且为128的倍数，1≤K≤min(32,E)。
下面的 16 卡跨机路径另限 EP32、W4A8（A8W4）或 A8W8 和实际 B=1～32。
各 rank 必须同序、同 B 调用；仅推理，不支持反向、空/不等长 B 或图捕获。
`mma_type="fp8xfp8"` 使用 A8W8 路由专家，不支持共享专家/timer；**下文 benchmark 固定测试 W4A8，不提供权重 dtype 开关**。

兼容入口 `mega_moe(x, ids, weights, w1, w2, buffer, ...)` 返回 `(y, per_call_counts)`。
接口签名参照 DeepGEMM `78b6900`。本实现使用 int32 路由 ID、E8M0 输入 scale，
输入 buffer 是普通 NPU 存储，需要打包进入通信区。路由权重在 GMM2 后施加，
与在中间量化前施加权重的实现不保证数值等价；CPU 参考遵循本实现的计算顺序。

### 共享专家策略

当 W4A8 的 S>0 时，由实际 B 选择策略，容量 `num_max_tokens_per_rank` 不参与判断：

| 项目 | B<=256 | B>256 |
| --- | --- | --- |
| 输入 | 双 AIV Quant → AIV 同步 → Route → reset → 共享输入准备 | 双 AIV Quant → reset → 共享输入准备 |
| 本卡全核屏障后 | AIC/AIV0 共享 GMM1，AIV1 count/prefix/Dispatch | AIC/AIV0 共享 GMM1，AIV1 Route/子组及跨卡同步/count/Dispatch |
| 共享 GMM1 | 完整 gate/up 输出 N192 独立任务 | N256 gate/up 配对任务 |
| 共享 GMM2 | N224 | N256 |

两条路径均使用 CATLASS 后端、M256 和 N256 共享 SwiGLU。路由专家在 B≤256 使用逻辑 N128/物理 N256 融合；B>256 使用 gate256/up256 配对任务。
Token Combine 按实际单卡 B 选择：B≤256 使用六槽，B>256 使用两槽，与是否开启共享专家无关。
小 BS 使用逐槽 epoch/count 就绪协议，每个物理 AIV 本地推进一次 epoch，
输出握手再推进一次；大 BS 使用输入/输出跨卡握手。共享 GMM1 后无需全核屏障。
共享 SwiGLU 在 AIV1 上优先于 Combine，共享 GMM2 按本核路由完成及共享激活就绪推进。
S=0 的 W4A8 小 BS 也使用上述量化完成/count 就绪协议；W8A8 使用输入/输出跨卡握手。
tiling ABI 为 232 字节。
CATLASS GMM 的布局和同步机制见 [GMM 说明](csrc/kernels/mega_moe/catlass_gmm/README.md)。

所有 rank 必须按相同顺序、相同 B 调用，切换 B 时在计时前集体调用 `prepare(B)`。
已有 `prepare` 一致性检查用于防止各 rank 选择不同握手策略。共享计算五类 timer 保留，
不添加 count 诊断事件；N192/N224 的 tile 元数据按实际 N 分块记录，JSON 格式不变。

验证边界建议覆盖 B=255/256/257/512、S=0/1/2，以及同一 buffer 下反复切换
`256 → 257 → 256` 和 epoch 回绕。CPU 协议测试不能替代 CANN 编译和 NPU 精度/性能验证。

## 2. 编译安装

W4A8 在 B≤256 时使用较小 wave、rail 路由中转和打包 count。
原有路径的 rail 按每8个 rank 一组，EP≤8 或完整8卡分组启用，其余 EP 回退直发路由。
显式指定 `ranks_per_node=16` 时 rail 使用16卡分组；跨机采用下文的 MTE＋RoCE 路径。
B>256 保留 Prefill 输入协议，所有路径保留输出跨卡同步。

要求 Linux、Python ≥3.11、Ascend950、CANN 9.2.0 及匹配的 Torch/torch_npu；框架环境由使用者准备。

```bash
git clone https://github.com/lueelu/DeepEP.git
cd DeepEP
source /path/to/cann/set_env.sh
bash experiments/megamoe/scripts/build.sh
python -m pip show ascend-deepep
```

在匹配的 Torch/torch_npu 环境中执行。实验构建入口默认开启 MegaMoE 和 timer，
自动初始化固定版本 SHMEM/CATLASS 子模块、校验或构建 SHMEM，再构建安装 wheel。
CATLASS 只拉取本仓使用的 C++ 头文件所在仓库，不递归下载 DSL 的 LLVM/Triton 或测试依赖。
`PYTHON=/path/to/python` 可指定解释器；`DEEPEP_MEGAMOE_TIMER=OFF` 关闭设备打点。
已激活 Conda 时，入口会在 Python 启动前优先使用该环境的运行库。
已有 SDK 可用 `SHMEM_ROOT`、`CATLASS_ROOT` 指定；已有子模块须保持干净且版本与[依赖锁文件](../../dependencies.lock.json)一致。
需要重建仓内 SHMEM 时，停止使用该 SDK 的作业后传 `--rebuild-shmem`。
wheel 位于 `dist/build-*/`；运行前按构建输出设置 SHMEM 的 `LD_LIBRARY_PATH`。
基础包的 `pip wheel` 默认关闭 MegaMoE，详见[构建文档](../../docs/build.md)。

两机各16卡需要按实际网卡构建 RoCE SDK。XSCALE 与 HNS_1825 二选一，例如：

```bash
# 已有默认 SHMEM 构建时，先停止使用该 SDK 的作业，再显式重建。
bash experiments/megamoe/scripts/build.sh --rdma-backend XSCALE --rebuild-shmem
# 使用 HNS 1825 网卡时，将 XSCALE 换成 HNS_1825。
```

首次构建可省略 `--rebuild-shmem`。后续复用仓内 SDK 时继续传入相同的 `--rdma-backend`。
默认值 `NONE` 保留不启用 RoCE 的 SDK 构建；16卡单机及原有8卡分组路径不要求 RoCE。
后端写入 `deepep-sdk.json` 并参与构建缓存和复用检查，MegaMoE 的 Host/Device 编译自动采用匹配配置。
外部 SDK 需由提供方登记真实后端，见[SDK 登记说明](../../docs/build.md#分步骤构建外部-sdk打包或-ci)。
所有运行节点安装相同 wheel、使用与 wheel 匹配的 SHMEM 库，并按部署环境配置 NPU—NIC 绑定与 RoCE 网络。
`DEEPEP_SHMEM_ENDPOINT` 是进程会合地址，不替代网卡 IP/GID 配置。

## 3. 多机测试

### BS32：保存配置，分别编译和运行

两台机器各16卡的 BS32 精度、性能及流水测试使用 `scripts/run_bs32.sh`，在**已有的同一个 checkout** 中执行。
此入口不重新克隆、不自动拉取代码，也不在运行时编译或安装；依赖留在当前仓库的 `third-party/`。
模型固定为 H=3584、I=1280、E=384、K=8、S=0、EP32，使用 FP8 输入和 FP4 权重、balanced 路由及全部 token 精度检查。

先激活匹配的 Torch/torch_npu Python 环境，在两台机器各配置一次：

```bash
bash experiments/megamoe/scripts/run_bs32.sh configure
# 依次输入本机编号（0/1）、节点0的IP、CANN set_env.sh、XSCALE/HNS_1825。
# 默认保存当前 Python、端口29654/26733、Golden workers=4。

# 两台分别编译安装；只在首次安装或需要更新构建时执行。
bash experiments/megamoe/scripts/run_bs32.sh build

# 两台分别执行，只运行测试。以后重复这条命令即可，不再询问配置。
bash experiments/megamoe/scripts/run_bs32.sh run
```

编译 warning 太多时使用 `bash experiments/megamoe/scripts/run_bs32.sh build --errors-only`。
终端只显示构建阶段提示、错误及其前6行/后8行上下文；完整输出（包括 warning）仍写入
`<case_dir>/build_node_<rank>_<随机字符>/build.log`，成功或失败时都会打印日志路径。
这是终端显示选项，不改变编译器检查或失败退出码，也不会隐藏被提升为 error 的 warning。
默认 `build` 同样保存日志，并显示完整输出。屏蔽模式下没有持续输出时，可用 `tail -f <日志路径>` 查看进度。

配置默认保存到 `.megamoe-bs32/<hostname>.json`，已加入 Git 忽略规则；共享 checkout 的两台主机也各用自己的文件。
Python 使用保存的绝对路径，后续自动加载保存的 CANN 环境；迁移或删除 Python 环境后需显式更新配置。
可以用 `--config /absolute/path/node.json` 指定其他配置文件，之后各阶段都须传同一路径。
配置文件是 JSON 数据，不作为 shell 脚本执行；环境变量不会覆盖已保存的节点/IP设置。

```bash
bash experiments/megamoe/scripts/run_bs32.sh show-config
# 显式修改指定字段，其他设置保留；改节点0的IP时，两台都要修改。
bash experiments/megamoe/scripts/run_bs32.sh configure --master-addr 192.0.2.10
# 输出放到数据盘，避免每次在 /root 新建整个仓库。
bash experiments/megamoe/scripts/run_bs32.sh configure --case-dir /data/megamoe-bs32
# Python 环境变更时：
bash experiments/megamoe/scripts/run_bs32.sh configure --python /path/to/env/bin/python
```

也可直接编辑已保存的 JSON（路径使用绝对路径）。更改 CANN、Python 或 RoCE 后端后应重新执行 `build`；
更换仓内 SHMEM 后端时须停止使用该 SDK 的作业，再执行 `build --rebuild-shmem`。
`run` 自动生成本机16个 rank 的 CPU Golden，后续校验并复用缓存；可提前执行 `golden` 单独准备。
默认 Golden 位于 `benchmark-runs/bs32/golden/`，每次运行的配置快照和日志位于该目录上级的 `node_<rank>_<随机字符>/`。
配置与结果均保留在原目录，`git pull` 不会清除它们。此入口使用仓内 SHMEM/CATLASS，不接受外部 SDK 环境覆盖。
若之前使用的是复制到服务器的旧 `run_bs32.sh`，请改用上述仓库内路径，旧脚本不会随 `git pull` 更新。

精度通过后，直接复用上述机器配置测性能；两台机器都执行同一个命令：

```bash
bash experiments/megamoe/scripts/run_bs32.sh perf
# 可选：两台使用相同的预热、采样次数。
bash experiments/megamoe/scripts/run_bs32.sh perf --warmup 20 --iters 100
```

`perf` 不生成或读取 Golden，不重新编译。默认预热20轮、采集100轮，并统计后50轮；
模型、BS32、FP8输入、FP4权重、balanced路由均与精度用例一致。
沿用 benchmark 的前置对齐计算，临时占用约12 GB显存/卡；对齐计算不计入 MegaMoE kernel Duration。
结果保存到 `<case_dir>/perf_node_<rank>_<随机字符>/`，终端会打印完整路径和汇总：

| 文件 | 内容 |
| --- | --- |
| `summary.txt` | 全EP每轮最短/最长 rank 的平均 kernel Duration，单位微秒 |
| `case.json`、`launch.json`、`run.json` | 模型与轮次、机器配置、运行模式与选中的 timer ranks |
| `console.log` | torchrun 与 benchmark 完整输出 |
| `rank_<全局rank>/bs_32/profiling.json` | 全32 rank 的汇总及逐轮耗时 |
| `rank_<全局rank>/bs_32/kernel_samples.json`、`profiler/` | 本 rank 的全部原始样本及 profiler 数据 |

现有 benchmark 日志的 `E2E` / `mean_us` 是每轮**最短** rank 的平均值；
`rank-max-mean` / `rank_max_mean_us` 是每轮**最长** rank 的平均值，建议同时记录并查看 rank 分布。
它们是融合 kernel 的设备执行时长（含内部通信和等待），不含 Python 调用、权重准备或退出清理，
也不代表经过跨设备时钟校准的全局起止跨度。`perf` 不传入设备 timer；安装了支持 timer 的 wheel 也不会打点，
wheel 的编译配置可在每个 rank 的 `manifest.json` 中核对。需要恢复不支持 timer 的构建时执行普通 `build`。

采集内部流水时，先在两台机器分别构建安装支持 timer 的 wheel，再在两台机器启动 `trace`：

```bash
# 两台均安装完成后再开始 trace；无需重新构建 SHMEM。
bash experiments/megamoe/scripts/run_bs32.sh build --timer --errors-only
bash experiments/megamoe/scripts/run_bs32.sh trace

# 默认采集跨机配对 rank0、rank16；也可扩大覆盖范围，两台参数必须相同。
bash experiments/megamoe/scripts/run_bs32.sh trace --timer-ranks 0-31 --warmup 2 --iters 1
```

`trace` 独立采集设备流水，不执行性能 Profiler 或 Golden 校验。默认预热2轮、运行1轮，
每个 rank 共调用3次 MegaMoE；仅在采集批次的最后一轮、选中的 rank 上设备打点。
可用 `--warmup`、`--iters` 覆盖次数；采样前仍保留原有的 Exp 对齐计算，临时占用约12 GB显存/卡。
正式性能测试使用 `perf`，其默认预热20轮、采集100轮不变。
`trace` 不会自动构建；未安装支持 timer 的 wheel 时会在启动 worker 前提示执行 `build --timer`。
更新后需在两台机器重新执行上述 `build --timer` 安装新版 Python 包，才能使用独立的 trace 模式；SHMEM 可复用。
32个 rank 始终全部参加运行；采集后各节点只自动解析自己拥有的选中 rank，默认隐藏 WAIT 事件。
`trace --include-wait` 显示全部 WAIT，`trace --hide-wait` 显式隐藏。该开关只影响流水 JSON 和事件汇总，
原始 BIN/解析 CSV 仍保留 WAIT，其他事件的时间戳不变，不压缩等待时间。

A8W4 的跨机 RDMA 在所属 AIV 的 `/RDMA` 子轨道记录，带有阶段（`dispatch_meta`、
`dispatch_payload`、`combine`）、对端全局 rank、数据/commit 类型及字节数：

| 事件 | 含义 |
| --- | --- |
| `AIV_RDMA_POST` | 非阻塞 put 调用的提交时间；不代表数据在网络上的完整传输时间 |
| `AIV_RDMA_QUIET_WAIT` | 已有 QP quiet 调用的剩余完成等待；data 的 bytes=0 表示批量字节数未统计，commit 为4字节 |
| `AIV_RDMA_RECV_WAIT` | 已有 commit 等待及后续加载排序；包含等待对端生产、传输和发布 commit 的时间 |

默认只显示 POST；诊断通信等待时使用 `--include-wait`。设备打点不新增通信同步，仍有采样开销，
性能基线继续使用独立的 timer 关闭采样。新打点改变了原始 buffer 布局，两节点都需重新 `build --timer`，
并用同一版本解析器解析新采样；旧 BIN 请用采集时版本解析器。
已有的 `*_parsed.csv` 可直接重新出图（无需重跑设备，也可用于旧布局的已解析 CSV）：

```bash
python -I -m deep_ep_experimental.megamoe.timer.trace_parser \
  --timer-dir /实际结果目录/bs_32/timer --output-dir /实际结果目录/timer_hidden \
  --rank-start 0 --rank-end 0 --time-origin rank
```

上述命令默认隐藏 WAIT；加 `--include-wait` 可恢复显示。节点1的单卡采样对应 rank16，需同时修改两个 rank 参数。

排查 `AIC_WAIT_INPUT_PREP` 结束到首次 GMM 之间的空白时，两节点更新并重新 `build --timer`，
然后运行 `trace --include-wait`。新增的启动阶段位于各核 `/STARTUP` 轨道：

| 事件 | 测量范围 |
| --- | --- |
| `AIC/AIV_MOE_STARTUP` | 路由专家阶段入口到本核首次 Dispatch、GMM1 prologue 或 GMM1 tile 入口；无任务的核在阶段退出时结束 |
| `AIC/AIV_DISPATCH_BUFFER_INIT` | Dispatch 缓冲区视图和 scratch 地址初始化 |
| `AIV_INPUT_PREP_BARRIER_WAIT` | 输入准备后的既有 PIPE_ALL 排空和 AIC/AIV 屏障；AIC 对应原有 `AIC_WAIT_INPUT_PREP` |
| `AIV_COUNT_TABLE_PREPARE` | AIV1 准备 count/prefix 表的完整父区间 |
| `AIV_COUNT_METADATA_WAIT` | 获取所有来源的有效 count：包含 DMA、epoch 校验、转置、重试与退避，并非纯网络延迟 |
| `AIV_GATEWAY_PAYLOAD_CHECK` | 构造所需来源集合、检查本核负责的 gateway、发布并等待本卡就绪槽的父区间 |
| `AIV_GATEWAY_PAYLOAD_WAIT` | 唯一检查者获取所需 gateway 的 `RelayPayloadReady` 及发布前的 GM 顺序保障；显示 source rank、gateway rank 和 sequence |
| `AIV_COUNT_PREFIX_BUILD` | 计算专家 token 总数和 prefix 表，包含既有向量完成同步 |
| `AIV_COUNT_TABLE_PUBLISH` | count/prefix 写回、既有 MTE3 完成同步和 count-ready 发布 |
| `AIC/AIV_COUNT_READY_WAIT` | AIC/AIV0 等待对应 block 的 AIV1 发布 count-ready |

这些观测不新增 PIPE_ALL、跨核屏障或通信。`AIC/AIV_MOE_STARTUP` 是启动总区间，
不能与其子事件相加；首个 GMM tile 内部的输入等待仍由原 `AIC_GMM1_WAIT` 记录。
网关由 AIV1 分工检查，单条 WAIT 是检查者开始时的剩余等待，不能当作该来源的完整传输耗时。
其他核等待本卡就绪槽的时间只计入 `AIV_GATEWAY_PAYLOAD_CHECK`，不重复记作 gateway WAIT；
本卡消费者的父区间减去 gateway WAIT 仍包含集合构造和发布，不能当作纯本卡等待。
如果总启动区间长而以上子阶段都短，剩余部分包含专家扫描、调度及 GMM 调用内部首次 tile 前的准备。

解析同时输出 `startup_breakdown.csv`，按 rank/group/core 列出阶段起点、耗时及网关身份，
始终保留启动 WAIT，不受 JSON 的 `--include-wait` 开关影响。时间基准为每个 rank 的原始首事件；
`scope=envelope` 是父区间，各核和父子区间不可累计当作端到端耗时。
新启动打点再次扩展了 buffer 布局，旧 BIN 需使用采集时的解析器；旧采样无法补出此前未记录的阶段。

跨机 Combine 的推进开销在 `/COMBINE_PROGRESS` 子轨道单独记录：

| 事件 | 测量范围 |
| --- | --- |
| `AIV_COMBINE_INIT` | 首次构建私有 pending masks 的元数据 DMA 和标量处理；每次算子只记录一次 |
| `AIV_COMBINE_PROGRESS` | 初始化后的单次扫描和提交；记录进入/退出时待发的行数、POST 数、扫描槽位数及扫描/控制累计周期 |
| `AIV_HOST_COMBINE_BARRIER_WAIT` | 已有16卡机内完成屏障的完整调用，包含发布、轮询及本卡同步 |
| `AIV_COMBINE_DRAIN` | 屏障后最终 drain 的父区间，包含子 progress 和退避；记录进入/退出时待发行数 |

`scan_control_us` 包含游标扫描、ready flag 检查和私有 pending 状态维护，排除初始化、
`RocePutBytes` 调用和 drain 退避；它仍包含测量边界的开销，不能当作纯 GM 访存耗时。
progress/drain 的父区间不能与内部 RDMA 或子 progress 相加。初始化是独立区间，
已全部发送的 progress 快返不产生扫描记录；零待发的最终 drain 仍记录0行，便于确认完成。
这些观测不新增流水排空、跨核同步或通信，不以计时结果决定调度。

解析另输出 `combine_progress_breakdown.csv`，始终保留上述四类数据，即使 JSON 隐藏 WAIT。
其中 `pending_before` / `pending_after` 是行数；`posts` / `scanned_rows` 只属于 progress，
drain 行的这些字段留空。通过它比较单次推进耗时与最终剩余行数，判断缩减预算是否把工作移到了尾部。
完整 parsed CSV 的文本导出会保留新记录，恢复时自动重建此报告，无需额外附件。

包含 Combine 推进诊断的版本扩展了 timer buffer 布局：两节点均须重新 `build --timer` 并使用同版本解析器采集新 trace；
已有 SHMEM SDK 可复用。旧 BIN 使用采集时版本解析器；旧 parsed CSV 仍可读取，但没有新增观测。

结果位于 `<case_dir>/trace_node_<rank>_<随机字符>/`，保留配置快照、`console.log` 和各 rank 的资源信息，
不生成 `profiling.json`、`kernel_samples.json` 或 `profiler/`，流水输出包括：

| 文件 | 内容 |
| --- | --- |
| `summary.txt` | 流水采集次数、本机选中 ranks 及输出路径 |
| `parse_timer.log` | 本机流水解析日志 |
| `bs_32/timer/rank_<r>_npu_time.bin` | 原始设备打点，可重复离线解析 |
| `bs_32/timer/rank_<r>_npu_time.json` | 单 rank 的 AIC/AIV 流水，支持 Chrome JSON trace 查看器 |
| `bs_32/timer/all_rank_trace.json` | 本机选中 ranks 的合并流水，不包含另一台机器的原始数据 |
| `bs_32/timer/timer_breakdown.txt/json` | 阶段耗时拆分；本机自动解析后对应本机选中 ranks |
| `bs_32/timer/combine_progress_breakdown.csv` | Combine 初始化、逐次推进、机内屏障和最终 drain 的耗时及待发行数 |

流水按每个 rank 自己的起点归零，不能用不同 rank 的坐标相减计算跨机延迟。
设备打点反映代码阶段及等待，不是逐 WQE 的网卡执行轨迹；正式性能基线使用 `perf` 的结果。
`trace` 默认只采两张卡，以控制原始 timer 文件及解析开销；分析慢 rank 时可通过 `--timer-ranks` 定点采集。
运行、导出或自动解析失败均返回非零退出码，不打印成功的 `[DONE] trace`。

### 通过限长文本复制传输流水

服务器只能复制文本、每次最多65535字节时，使用离线工具 `experiments/megamoe/scripts/trace_text.py`。
只需更新仓库中的 Python 文件；已有采样可直接导出，**无需重编译、重新安装 wheel 或重跑算子**。
工具使用 Python 标准库，不需要 Torch、NPU 或额外压缩软件，也可在 Windows 上恢复。

先在每台服务器分别导出一个已完成的 trace 结果目录：

```bash
# 也可直接传 bs_32/ 或 bs_32/timer/；有多个 BS 时必须指定一个。
# output-dir 必须尚不存在，避免混入上次导出的分片。
python3 experiments/megamoe/scripts/trace_text.py export \
  /实际结果目录/trace_node_0_xxx \
  --output-dir /tmp/trace-share-node0 \
  --label "EP32-BS32-node0"

# 每次只显示一块；完整复制 BEGIN 到 END（含这两行），作为一条消息发送。
python3 experiments/megamoe/scripts/trace_text.py show /tmp/trace-share-node0 --part 1
# 按 export 报告的总块数继续 --part 2、--part 3……
```

默认导出目录中**所有已解析的 rank**，也可用 `--ranks 0,3-5` 缩小范围。每块默认不超过60000字节，
可用 `export --max-bytes 65535` 调整；上限计算包括编号、SHA-256、Base64、头尾标记、最终换行，
并按 CRLF 换行预留空间。`show` 的标准输出只有一块 ASCII 文本，不带日志或颜色。
不要一次打印所有分片，也不要把终端提示符、命令、`...` 或手工删改后的内容放进块内。
每块的完整校验信息都在文本中，**不必另行复制导出目录的 manifest.json**。

传输保留各 rank 的 `rank_<r>_npu_time_parsed.csv` **完整原始字节**，包括全部核、WAIT、
整数 cycle、wave/tile/expert、RDMA 和网关元数据；不裁时间、不抽样、不根据界面的 WAIT 隐藏设置删事件。
整个包先 XZ 无损压缩，再 Base64 编码和分片。导出会检查 CSV 的核覆盖、列数和任务计数，
拒绝明显截断的文件。输出块数取决于实际数据，不保证完整流水只需一块。

标准结果目录下的 `case.json`、`run.json`、所选 rank 的 `manifest.json` 和 `resources.json`
会一起打包。额外的性能结果可显式附带，例如：

```bash
python3 experiments/megamoe/scripts/trace_text.py export /实际结果目录/bs_32/timer \
  --output-dir /tmp/trace-share-with-perf \
  --attach /对应perf结果目录/rank_0/bs_32/profiling.json
```

默认不收集 `launch.json`、环境变量或其他日志。两个节点分别生成独立包，不要把不同节点、不同导出或
不同次运行的分片合并为同一个包；label、rank 列表和运行 manifest 用于识别各自的采样。
只会传输实际采集的卡，导出不能补出未打点的 rank。

接收端把复制的文本存入 UTF-8 文件：可以每块一个文件，也可以在同一个文件中连续粘贴多个完整块。
可以乱序；重复粘贴内容完全相同的块不会影响恢复。检查和恢复示例：

```bash
# 检查已收到多少块、缺哪些编号；不写恢复文件。
python3 experiments/megamoe/scripts/trace_text.py inspect copied-messages.txt

# 校验全部分片、整包和每个原文件后，恢复到一个新目录。
python3 experiments/megamoe/scripts/trace_text.py restore copied-messages.txt \
  --output-dir restored-trace-node0

# 也可传多个文本文件，或用单独的 "-" 从 stdin 读取。
python3 experiments/megamoe/scripts/trace_text.py restore part1.txt part2.txt \
  --output-dir another-restored-trace
```

缺块、冲突重复、Base64 损坏、SHA-256 不符或混包时会明确报错。CRLF 和 Base64 的普通空白/换行变化可接受，
不会尝试猜测修复其他字符。恢复目录必须尚不存在，已有文件不会被覆盖。
恢复后 `timer/` 中包含逐字节一致的 CSV，并自动生成保留 WAIT 的单 rank/合并 JSON、
`startup_breakdown.csv` 和阶段拆分；使用 `restore --no-render` 可只恢复文件。
附带配置位于 `context/`，显式附带文件位于 `attachments/`。
**跨 rank 的时间仍各自归零，不能据此把两张卡的坐标差当作机间延迟。**

已安装新版 Python 包时，也可通过
`python -I -m deep_ep_experimental.megamoe.timer.text_transfer` 使用相同子命令。

### 两机各16卡：MTE＋RoCE Decode

通信模式由已有的 `ranks_per_node` 决定，没有新增公开 `transport` 参数：

| 配置 | 通信路径 |
| --- | --- |
| `ranks_per_node=8` | 保留原有 MTE 路径 |
| `ranks_per_node=16`、EP16、同一主机 | MTE |
| `ranks_per_node=16`、EP32、两台主机 | 机内 MTE，跨机 RoCE |

`ranks_per_node=0` 保留原有默认行为，不自动检测16卡平台。两机路径要求一台机器对应 rank 0～15、
另一台对应 rank 16～31，gateway 配对为 `0↔16`、`1↔17`、…、`15↔31`。
初始化会集体验证 Linux 主机身份、设备唯一性、rank 分组、SDK 后端及库身份；不匹配即报错。
两机路径支持 `mma_type="fp8xfp4"`（W4A8/A8W4）或 `"fp8xfp8"`（A8W8），实际每卡 B=1～32；
FP8 和内部量化的 BF16 输入均沿用原接口，容量可大于实际 B。B>32 及更多16卡主机暂不支持。
A8W8 当前只运行路由专家，`num_shared_experts` 必须为0，不能传入 timer。

两台机器的所有 rank 都固定由物理 AIV0 打包本地元数据，物理 AIV1 发送 RDMA。
AIV1 在 Dispatch 和 Combine 全程独占 QP1，包括 Put、quiet 和 commit；AIV0 不推进 RDMA。
此分工适用于 A8W4/A8W8；host_id 仍用于选择 rail 槽、gateway 和接收转发分工，
打包使用 AIV0 自己的 epoch 计数槽，发送前仍等待 `LocalRailPacked`。

跨机输入 payload 在本卡全部 token/scale 准备完成后，按实际 B 一次 Put 到配对 gateway，
不再扫描路由并拆成多个连续 token 段；B 小于预分配容量时只发送实际 B 行。
接收区保留原 token 编号，专家仍按路由元数据读取，多传的行不会增加计算。
该策略用于上述 B≤32 的 A8W4/A8W8 路径，包括没有远端路由的批次。
元数据仍先完成并发布通知；payload 完成等待仍延后到 owner 的 reset/shared-input 工作之后，
保留 quiet、commit、commit quiet 和对端 commit 等待顺序。
H=3584、B=32 时 payload 为118784 B；相较已测 EP32/K8 balanced 样本的8次7424 B提交，
该策略以每 rank 额外59392 B流量换取一次提交和更少的发送端扫描工作。
性能验证应同时检查 payload POST 次数、quiet、输入准备结束及首 GMM 时间，
并以关闭打点的 `rank-max-mean` 判断整体收益，不能仅由 POST 次数推断加速比例。

跨机输入的 gateway 依赖按 AIV1 block 分工：各核从同一份有效 count 表得到本 rank
实际需要的来源；唯一检查者确认对应 gateway 就绪后，向本卡独立的512 B槽发布当前 epoch，
所有消费者等到所需本卡槽就绪后才发布 count-ready。各检查者先完成自己的检查和发布，
再消费本卡通知，避免把并行检查串行化。28个 AIV1 都需要16个来源的用例中，
gateway 检查调用由448次降为16次；这些是依赖检查调用数，未就绪时仍会轮询。
新增16×512 B槽及对齐 padding 纳入现有输入屏障前的 workspace reset，
不改变 peermem 通信布局，也不新增全核屏障。转发前显式执行
[`DataSyncBarrier<DDR>`](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/sync_control/intra_core_sync/DataSyncBarrier_ISASI.md)，
消费者保留 scalar→MTE2 顺序保障。A8W4/A8W8 共用此协议。

A8W4 的跨机 Combine 在 AIV1 的计算间隙分批推进：两台机器均每次最多提交2个 Put、
扫描64个 `[token, top-k]` 槽位。扫描游标跨调用保留，已发送槽位不再读取 ready flag；
全部发送后立即返回。推进穿插于 AIV1 的 activation、dispatch 和本地 Combine 工作，
最终 host barrier 后完整 drain，再执行 payload quiet、commit、commit quiet 和对端 commit 等待。
单次 Put 内部仍可能因队列回压等待，这些计数预算并非硬性时间上限。
扫描在单次调用内缓存当前 token 的 pending mask，通过递增槽位推进，避免逐行除模；
连续发送段在 Put 返回后按 token 清除对应位。扫描仍逐槽计数，保留洞、游标绕回、
预算截断和相邻行合并的原有语义，不减少推进机会。

Combine 路由首次初始化使用 owner 专用256 B UB 分块 DMA 读取已接收的 rail 元数据；
BS32、K8 时为5次256 B DMA，跳过每个512 B packet 的末尾 marker。
该初始化优化也适用于 A8W8，其 Combine 仍由 AIV1 每次完整扫描。
更新后两台均需重新构建安装；按上述 BS32 流程先执行 `run` 校验精度，再执行
`perf` 比较 `rank-max-mean`，并固定 `trace --timer-ranks 0,14,16,30 --include-wait`
检查输入 payload、计算尾部、Combine POST 分布和 quiet/commit 等待。CPU 回归不能替代这些设备检查。

两机 RoCE runtime 的 `barrier()` 和 `destroy()` 先同步本卡全部 stream，再通过现有
StoreGroup（TCPStore）或 ProcessGroup 交换完成状态；全部 rank 成功后才允许清理 SHMEM。
当前 kernel 在返回前已等待应用 QP1 完成，这一顺序确保释放前没有未完成的通信。
这里不调用直接读取对端 GM 的通用 `aclshmem_barrier_all()`，也不要求机内各 rank 建立 RoCE QP。
所有 rank 必须按相同顺序调用，调用期间不得并发提交新的算子；任一 rank 同步失败则保留资源并要求重启作业。
该修复涉及 Python 和原生扩展，两台机器都需重新构建/安装匹配的 wheel；已有 SHMEM SDK 无需为此修改。

在两台已分配设备的机器上分别执行下列命令，只将 `NODE_RANK` 改为0或1。
地址和端口是占位示例，需替换为实际配置；两台机器各启动16个进程：

```bash
export MASTER_ADDR=192.0.2.1
export MASTER_PORT=29654
export DEEPEP_SHMEM_ENDPOINT=tcp://192.0.2.1:26733
export NODE_RANK=0  # 另一台机器使用1。
torchrun --nnodes=2 --nproc-per-node=16 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" --max-restarts=0 \
  --module deep_ep_experimental.megamoe.example --input bf16
```

示例使用 B=17，并从 `LOCAL_WORLD_SIZE` 传入 `ranks_per_node=16`。它检查输出有限性，
该 example 以及现有 benchmark 均固定使用 W4A8，没有 `--mma-type` 开关。
W4A8 的完整精度比较需要 benchmark 的 CPU Golden。共享存储上先生成所需 case 的 Golden，再在两台机器启动：

```bash
# 仅生成一次；参数须与后续 benchmark 完全一致。
python -m deep_ep_experimental.megamoe.golden --config deepseek_v4_pro \
  --bs-values 1,17,31,32 --ep-size 32 --ranks 0-31 --input bf16 --routing balanced \
  --golden-dir /path/to/shared/golden

# 每台机器分别运行，沿用上面的 MASTER_*、NODE_RANK 和 SHMEM 环境。
torchrun --nnodes=2 --nproc-per-node=16 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" --max-restarts=0 \
  --module deep_ep_experimental.megamoe.benchmark --config deepseek_v4_pro \
  --bs-values 1,17,31,32 --mode accuracy --input bf16 --routing balanced \
  --golden-dir /path/to/shared/golden --output-dir /path/to/new/accuracy-run
```

性能测试将 `--mode accuracy` 改为 `--mode perf`，并选择新的输出目录；计时范围仍遵循下文说明。
还应覆盖 FP8 输入、random/skewed 路由、S=0/1、重复调用及 B 切换。
本分支的真实 CANN 编译、两机 NPU 精度、RDMA 到达可见性和性能验收仍待完成；CPU 测试不代表设备通过。

A8W8 使用公开 `mega_moe` 接口。下面片段假定32个进程已设置各自 NPU、endpoint 和 `group`，
模型已提供同一 NPU 上的连续输入、路由及原始 FP8 ND 权重：

```python
from deep_ep_experimental.megamoe import SymmBuffer, mega_moe, transform_weights_for_mega_moe

B = 32
# x_bf16: BF16 [B,H]；ids: int32 [B,K]；weights: FP32 [B,K]。
# E 为全局路由专家数，L=E/32。
# raw_w1: (FP8 E4M3 [L,2I,H], E8M0 [L,2I,H/64,2])。
# raw_w2: (FP8 E4M3 [L,H,I], E8M0 [L,H,I/64,2])。
buffer = SymmBuffer(
    group, num_experts=E, num_max_tokens_per_rank=B, num_topk=K,
    hidden=H, intermediate_hidden=I, num_shared_experts=0,
    mma_type="fp8xfp8", ranks_per_node=16,
)
w1, w2 = transform_weights_for_mega_moe(raw_w1, raw_w2)
buffer.topk_idx[:B].copy_(ids)
buffer.topk_weights[:B].copy_(weights)
buffer.validate_inputs(B, x=x_bf16)
buffer.prepare(B)
y, per_call_counts = mega_moe(
    x_bf16, buffer.topk_idx[:B], buffer.topk_weights[:B], w1, w2, buffer,
)
# 所有 rank 成功完成后，同序销毁；异常通知/退出按 example.py 的处理方式。
buffer.destroy()
group.close()
```

A8W8 的完整数值与性能验证需使用匹配 FP8 权重的模型参考，不能将上述 W4A8 Golden 当作 A8W8 结果。

### 原有每机8卡：配置与启动

下面的 `batch_launch.py` 仍按每机8个进程组织作业；两机16卡使用上面的 `torchrun` 命令。
所有节点需要相同绝对路径的共享 checkout、匹配的 Conda/CANN 环境、SSH 访问和每机8张空闲 NPU。
在仓库外创建机器配置，通过 `--hosts-file` 显式传入。以下地址仅用于文档示例，需替换为已分配的节点：

```json
{
  "hosts": ["192.0.2.1", "192.0.2.2", "192.0.2.3", "192.0.2.4"],
  "master": "192.0.2.1"
}
```

EP32/64/128 分别需要4/8/16个不同节点。`master` 可省略，默认使用列表首节点。
主节点排在 rank0～7，其余节点保持列表顺序。仓库不提供实际机器池。
在一台控制机设置远端环境后启动：

```bash
export CONDA_BASE=/path/to/conda
export CONDA_ENV=your_env
export ASCEND_ENV=/path/to/cann/set_env.sh
export HOSTS_FILE=/path/outside/checkout/hosts.json
# REMOTE_USER 默认为控制机当前用户，可按需覆盖；默认使用 SSH 密钥。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --ep-size 32 --plan
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --ep-size 32
```

入口编译一次、逐节点安装同一 wheel，再运行测试及解析 timer；不会更新 Git 分支。
`CONDA_BASE`、`CONDA_ENV`、`ASCEND_ENV` 必须显式设置，路径对应每台远端机器。
密码登录可设置 `SSH_AUTH=password` 并安装 sshpass，密码通过交互输入或 `SSHPASS` 提供。
实际机器配置、凭据和输出报告不要加入版本控制。

### 常用参数

| 参数 | 默认 / 作用 |
| --- | --- |
| `--mode perf\|accuracy` | 默认 perf 执行性能测试；accuracy 检查全量输出和专家计数 |
| `--input fp8\|bf16` | 默认 fp8；只改变输入类型，权重仍为 FP4 |
| `--config NAME或路径` | 默认 `deepseek_v4_pro`；可传模型 JSON 路径 |
| `--bs-values 96,4096` | 覆盖 JSON 中 BS 列表；每个 BS 单独启动 |
| `--routing balanced\|random\|skewed` | 默认 balanced；random/skewed 用于精度压力测试 |
| `--no-timer` | 关闭打点和解析；默认自动保存、解析主节点 rank0～7 |
| `--no-parse-timer` | 跳过自动流水解析，保留原始 timer 和耗时拆分 |
| `--ep-size 32\|64\|128` | 默认64；分别为4/8/16台八卡机器 |
| `--hosts-file 路径 / --master-ip IP` | 必填机器列表 JSON / 覆盖列表内的主节点 |
| `--master-port / --shmem-port` | 默认29654 / 26733，必须不同且未占用 |
| `--skip-build / --wheel 路径` | 复用已安装身份匹配的唯一 wheel / 指定共享 wheel；不再编译，但仍向各机安装 |
| `--prepare-only / --plan` | 只部署并检查 / 只预览 |
| `--timeout 秒` | 每阶段、每 BS 默认7200秒 |

`balanced` 路由按全局 `[token, top-k]` 顺序轮转目标 rank，再轮转卡内 expert。
精度使用相同路由生成 CPU Golden。修改算子或包内 Python 后，需要重新构建安装，
首次运行时省略 `--skip-build`；修改文档或启动脚本无需重编算子。

以下命令使用8节点机器文件，执行 EP64：

```bash
# 全量精度测试，关闭设备打点。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --mode accuracy --bs-values 96,4096 --no-timer

# BF16 输入，性能 + 自动流水解析。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --input bf16 --bs-values 96

# EP128：将 HOSTS_FILE 改为包含16个节点的配置。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --ep-size 128
```

### 模型和共享专家配置

默认模型：H=7168、I=3072（gate+up 总宽6144）、全局384路由专家、TopK=6、S=0。
EP32/64/128 分别每卡12/6/3个路由专家；共享专家不计入384，`num_shared_experts` 表示**每卡**数量。

已提供两个配置，只有 `num_shared_experts` 不同，其余模型、BS、seed 和轮次一致：

- [`deepseek_v4_pro`](python/deep_ep_experimental/megamoe/configs/deepseek_v4_pro.json)：无共享专家，默认配置。
- [`deepseek_v4_pro_shared1`](python/deep_ep_experimental/megamoe/configs/deepseek_v4_pro_shared1.json)：每卡额外1个共享专家，不减少原有路由专家。

```bash
# 每卡1个共享专家，全量精度检查 BS96/4096。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --config deepseek_v4_pro_shared1 --mode accuracy --no-timer

# 使用相同配置执行性能测试，并自动采集、解析主节点8卡流水。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --config deepseek_v4_pro_shared1
```

需要其他数量可复制 JSON、修改 `num_shared_experts` 后使用 `--config /path/to/case.json`。
脚本自动生成对应共享权重、传给公开接口，CPU Golden 也包含共享专家；
**没有 `--shared-experts` 开关**。warmup/iters/seed 同样在 JSON 中修改。

### 测试结果

精度测试以相同输入、权重和路由生成的 CPU Golden 为参考，生成输出和专家计数的校验报告。
CPU Golden 缓存位于 `benchmark-golden/`，相同配置可复用。
性能测试根据关闭 timer 时 profiler 采集的算子 kernel Duration，生成耗时统计报告。
流水分析根据独立采集的设备 timer 数据，生成流水 JSON 和阶段耗时报告，默认解析主节点 rank0～7。

终端输出各 BS 的 `[PERF]` 耗时或 `[ACCURACY SUMMARY]` 精度摘要，结果保存至：
`benchmark-runs/ep<EP>_<时间>_<编号>/`。

| 目录内文件 | 内容 |
| --- | --- |
| `summary.txt` | 各 BS 的性能/精度汇总 |
| `logs/node_<n>_<stage>*.log` | 各节点的构建、Golden、运行、解析日志 |
| `bs_<B>/rank_0/bs_<B>/accuracy.json` | 全部 rank 的精度报告 |
| `bs_<B>/rank_<r>/bs_<B>/profiling.json` | 性能汇总；同目录保留 `kernel_samples.json` 和 `profiler/` |
| `bs_<B>/bs_<B>/timer/` | 主节点8卡原始 `.bin`、`rank_<r>_npu_time.json`、CSV 和 `timer_breakdown.txt/json` |

原始 timer 数据也可离线解析：

```bash
python -m deep_ep_experimental.megamoe.timer /实际结果目录/bs_96/bs_96/timer --ranks 0-7
```

所有 BS 执行成功后，终端输出 `[DONE] all nodes completed`；执行或报告校验失败时返回非零退出码。

## 4. 测试与验证范围

```bash
python -m pip install -r experiments/megamoe/tests/requirements.txt
python -m pytest experiments/megamoe/tests -q
```

保留公开 API、数值参考、Golden 缓存、打包、主机协调与计时解析测试。
CPU 检查不启动设备 kernel，也不作为设备验证结果。Ascend950 设备验证记录按以下维度区分：

- MegaMoE 开启/关闭、timer 开启/关闭的构建安装验证。
- BF16/FP8 输入、不同路由、共享专家、容量边界和多次调用的精度验证。
- 多流生命周期、BS 切换、初始化/销毁和多机长时间运行验证。
- 各声明 EP 规模的设备测试及同输入、同软件版本、同计时范围的性能对比。
