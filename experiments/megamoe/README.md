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
HNS1825 新固件适配固定到 SHMEM !944 合入版本；当前分支重建与历史 commit 兼容迁移见
[SHMEM !944 复测说明](../../docs/shmem-944.md)。
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
# 旧方法：整批 Exp20 + 连续 MegaMoE（默认）。
bash experiments/megamoe/scripts/run_bs32.sh perf --perf-method legacy
# 新方法：每轮清理 L2 + device sleep + SHMEM 全卡同步 + MegaMoE。
bash experiments/megamoe/scripts/run_bs32.sh perf --perf-method shmem
# 可选：两台使用相同的预热、采样次数。
bash experiments/megamoe/scripts/run_bs32.sh perf --warmup 20 --iters 100
```

`perf` 不生成或读取 Golden，不重新编译。默认预热20轮、采集100轮，并统计后50轮；
模型、BS32、FP8输入、FP4权重、balanced路由均与精度用例一致。
`--perf-method legacy` 保留连续批次方法：整批开始前通过 TCPStore 会合，排入一次 Exp 前置计算，
随后连续下发 MegaMoE。`alignment.repeats` 默认从800回退为20，复用约12 GB显存/卡；
轮间不插入同步或额外计算。`iteration_sync=none`，`alignment_scope=once_before_batch`。

`--perf-method shmem` 参考 [DeepGEMM-Ascend bench_msprof](https://github.com/deepseek-ai/DeepGEMM-Ascend/blob/main/deep_gemm/testing/bench.py)：
每轮写零一个大小为设备 L2 两倍的缓冲区，排入20,000,000系统计数周期的设备 sleep，再执行 SHMEM WORLD barrier，
最后下发 MegaMoE。这个模式不分配12 GB Exp缓冲区。单机调用 `aclshmemx_barrier_all_on_stream`；
跨机 RoCE 用单 AIV 辅助 kernel 调用 `aclshmemx_roce_barrier_all`，显式提供256 B UB和event 0，
使用 SDK 的全连接 QP及 WORLD 同步状态。当前锁定 SDK 的通用流上 barrier 通过远端 GM 同步，不能直接替代 RoCE 接口。
`iteration_sync=shmem_world_on_stream`，`alignment_scope=before_each_iteration`。

两个模式都只在批次边界做 Host 同步，不建立 HCCL 进程组。SHMEM barrier 在当前流排队，
MegaMoE 的专属流通过已有事件依赖等待它；下一轮预处理也等待前一轮完成。
只统计 MegaMoE kernel Duration，清 L2、sleep、barrier、Exp均排除在外。
`manifest.json` / `profiling.json` 记录 `perf_method`；每个 BS 的 `measurement.json` 还记录实际接口、L2缓冲字节数和sleep周期，
`shmem` 模式将不使用的 Exp配置记录为 `alignment=null`。两台机器必须选择相同方法，否则在开始采样前报错。
新模式需要两端重新 `build` 安装 wheel（无需为此重建相同版本的 SHMEM SDK）；首次 helper调用在采集外预热。
逐轮 barrier 用于减少快慢卡累积偏移，不保证各卡同一物理时刻启动。两种模式的缓存状态不同，性能结果应分别比较。
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

#### BS32：Dispatch 单次完成通知，双机 A/B（历史实验）

此实验已完成，保留用于复现 `546182b`；后续优化使用 `build`、`run`、一次 `perf` 的流程，
将结果与前一版本比较。当前 Combine owner 分工已变化，不满足此历史实验的单文件差异约束。
复现时使用相同的 `546182b` checkout，保留现有 BS32 配置及已构建的依赖，再分别执行：

```bash
bash experiments/megamoe/scripts/run_bs32_dispatch_ab.sh
```

脚本只构建两版 timer-OFF wheel：A 使用 `732bb3f` 的 Dispatch（metadata/payload 分别通知），
B 使用当前提交的 Dispatch（两次数据 Put 共用一次完成通知）。两版其他原生源码完全相同，
且须与 `732bb3f` 相同；当前实验仅允许 Dispatch 头文件存在差异，避免混入其他优化。
每个版本首次运行前执行全 token 精度检查，默认随后按 `A B` 各测一次，
共2次性能测试；每次预热20轮、采样100轮、统计后50轮，用于快速筛选优化方向。
如需复核运行间波动，可加 `--cycles 2` 按 `A B B A` 各测两次；
`--cycles 6` 才会执行原来的 `ABBA` 三遍。重复次数不影响两次构建和每版一次精度检查。
两台分别沿用自己的环境，不比较 CMake、SDK 指纹或编译选项是否相同。

A 首先构建，结束后恢复 **A（timer OFF）**；中途失败也在 A 已构建成功时尝试恢复。
在该历史提交上，普通 `run_bs32.sh build` 构建的是 B；结束后的已安装版本应以恢复状态为准。
结果目录仍是 `<case_dir>/ab_node_<本机编号>_<随机字符>/`；终端会打印可直接复制的 `cat` 命令，
两台各回传 BEGIN 到 END 的完整文本块，每块最多60000字节。摘要中 `B-A` 为负表示优化变快。
`META.experiment=dispatch`、版本的 `dispatch_commits` 和源码哈希用于区别上一轮实验。

支持原有 `--config`、`--cycles`、`--warmup`、`--iters`、`--control-port` 和 `--stage-timeout`。
已有两轮实验的结果都可用以下命令重新导出；脚本从结果内的计划识别实验类型，不会重新构建或运行：

```bash
bash experiments/megamoe/scripts/run_bs32_dispatch_ab.sh --export-text /absolute/path/ab_node_0_xxxxxxxx
```

#### BS32：上一轮调度改动的四组 A/B（历史实验）

此入口保留用于复现 `732bb3f` → `bfb6fe8`，要求内核仍为 `bfb6fe8`（例如 checkout `e888921`）；
Dispatch 的历史实验见上节。两台服务器保留现有 BS32 配置、
已构建的 SHMEM SDK 和 CATLASS，在相同提交、没有已跟踪文件修改的 checkout 中分别执行同一条命令：

```bash
bash experiments/megamoe/scripts/run_bs32_ab.sh
```

不用手动切换 Git 分支、修改内核或配置 SSH。脚本从当前提交导出独立源码副本，
只替换以下两项改动涉及的文件；当前 checkout 的内核须仍与 `bfb6fe8` 一致。

| 版本 | 激活优先调度 | Combine 发完后提前结束扫描 |
| --- | --- | --- |
| A | 关闭 | 关闭 |
| B | 开启 | 关闭 |
| C | 关闭 | 开启 |
| D | 开启 | 开启 |

两台分别构建四份 timer-OFF wheel 后缓存，在 `A B C D D C B A` 顺序上重复三遍，
每个版本测六次，每次预热20轮、采样100轮、统计后50轮；各版本首次运行前执行全 token 精度检查。
Golden 复用原配置的 `case_dir/golden/`。两端在构建、安装、精度和性能阶段同步，
核对测试源码和顺序、各机实际安装的包以及本机各 rank 的构建信息；发生错误、断连或超时后终止当前子进程。
直接沿用两台机器各自的构建环境，CMake 版本、SDK 指纹、搜索路径及编译选项不做跨机一致性检查，
各机完整构建信息保存在对应的 `builds/*/wheel.json` 中。
`CMAKE_PREFIX_PATH` 中完全相同的重复目录会按首次出现顺序去重后用于构建。
原始设置保存在 `build_settings_raw.json`，实际使用的设置保存在 `build_settings.json`。
脚本会安装测试 wheel，因此运行期间应独占这套 Python 环境和32张卡。结束时重新安装 D 组 timer-OFF wheel，
若中途失败则在 D 已构建成功的情况下尝试恢复；恢复状态写入 `status.json`。

协调端口默认是保存的 `master_port + 1`（默认 **29655/TCP**），需允许节点1连接节点0。
原有 torchrun/SHMEM 端口保持不变。两端应在10分钟内启动；每个阶段最多等待两小时。
需要修改时，在两台使用相同参数，例如：

```bash
bash experiments/megamoe/scripts/run_bs32_ab.sh --control-port 29656 --stage-timeout 14400
# 仅检查流程能否跑通：每组两次，不作为最终回退结论。
bash experiments/megamoe/scripts/run_bs32_ab.sh --cycles 1
# 若使用自定义机器配置，两台分别指定各自的文件。
bash experiments/megamoe/scripts/run_bs32_ab.sh --config /absolute/path/node.json
```

输出在 `<case_dir>/ab_node_<本机编号>_<随机字符>/`，不覆盖历史结果：

| 文件 | 内容 |
| --- | --- |
| `summary.txt` | 各组两种耗时的中位数/范围；B−A、C−A、D−A及交互项的成对差值 |
| `results.csv`、`results.json` | 每次独立测试的版本、源码指纹、耗时与结果位置 |
| `comparison.json` | 各组统计和每个正序/倒序块的差值；正值表示变慢 |
| `plan.json`、`build_settings.json`、`builds/` | 提交、两文件哈希、编译选项、wheel 信息和构建日志 |
| `runs/` | 精度/性能日志、32 rank 汇总和本机16 rank 的原始样本及 profiler |
| `status.json` | 完成/失败原因、最终安装状态 |
| **`copy_text_*/part_001.txt` 等** | 默认回传方式：可读精简文本，每块最多60000字节（包含头尾标记及 CRLF 换行） |
| `results.zip` | 服务器本地完整存档，含统计、配置、日志和样本；不含大体积源码/wheel/原始 profiler |

完成后，按终端最后打印的 `cat /.../copy_text_.../part_001.txt` 命令，
复制 **BEGIN 到 END 的完整文本块**发送即可。两台各复制自己的结果；每次只复制一个块，
不要一次 `cat` 所有分块。默认24次测量通常各一块；增加轮数超出容量时会自动分块，不丢弃测量记录。
每块按 UTF-8 字节计算，预留 Windows 换行扩展，严格小于65535字节，并包含块编号、整份报告及本块 SHA-256。

精简文本保留场景/轮数、节点和版本指纹、恢复状态、每次两种耗时和32个 rank 的平均耗时、
分组统计及成对差值；发生失败时保留受限长度的错误及日志摘要。逐次耗时/各 rank 均值精简到0.001微秒，
全精度数据和完整日志留在服务器。两台的32 rank汇总是同一份采样，统计时不会将其当作两次独立实验。

已经运行过旧脚本的结果可以离线导出，不需要重跑、编译或连接另一台机器：

```bash
bash experiments/megamoe/scripts/run_bs32_ab.sh --export-text /absolute/path/ab_node_0_xxxxxxxx
```

每次导出创建新的 `copy_text_*` 子目录，保留之前的结果。脚本保留逐次数据，失败也导出/存档已有结果；
只有完整跑完并恢复成功才标记完成。它不自动把一次慢测或差值为正判定为稳定回退。
若进程被 `SIGKILL` 或机器掉电，无法执行恢复与打包；已落盘数据保留，可先运行普通 `run_bs32.sh build` 恢复环境。

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
可用 `--warmup`、`--iters` 覆盖次数；与 `perf` 一样支持 `--perf-method legacy|shmem`，默认每批 Exp20 后连续下发 MegaMoE，
临时占用约12 GB显存/卡。未选中 timer 的 rank 也必须执行相同轮数。
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
| `AIV_GATEWAY_PAYLOAD_CHECK` | count-ready 发布后，按保存的来源集合检查本核负责的 gateway、发布并等待本卡就绪槽的父区间 |
| `AIV_GATEWAY_PAYLOAD_WAIT` | 唯一检查者获取所需 gateway 的 `RelayPayloadReady` 及发布前的 GM 顺序保障；显示 source rank、gateway rank 和 sequence |
| `AIV_COUNT_PREFIX_BUILD` | 从原始 count 保存 gateway 依赖集合，再计算专家 token 总数和 prefix 表，包含既有向量完成同步 |
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

头开销细分版本在 `/RELAY` 轨道新增 `AIV_RELAY_STEP`，包含 `packet_wait`、`topk_load`、
`route_build`、`route_post`、`route_complete`、`count_publish` 六类步骤，附带原 source rank 和专家范围。
其中 `route_post` 测量 DMA 提交及相关标量工作，`route_complete` 测量既有完成等待；二者不是链路传输时间。
`count_publish` 是 gateway 向专家卡发布计数，与本卡的 `AIV_COUNT_TABLE_PUBLISH` 含义不同。
JSON 隐藏 WAIT 时也隐藏 `packet_wait` 和 `route_complete`；`startup_breakdown.csv` 始终保留这些步骤。

`AIV_COUNT_METADATA_WAIT` 的 task/args 及 `startup_breakdown.csv` 新增快照次数、重试次数、快照字节数、
DMA/校验转置/退避累计耗时、成功快照耗时，以及最后失败快照的未就绪 source 列表和代表性本卡局部 expert 编号/epoch。
这些字段只覆盖 DMA 快照路径；不具备诊断数据的旧采样或逐槽回退路径留空，不作为0次重试解释。
每次成功获取只写一次诊断汇总。UB 空间足够时交替使用两份现有空闲 Dispatch 缓冲，保留最后失败快照；
成功后用一次 MTE3 批量复制到 timer GM，由主机解析 source/expert/epoch，设备不再逐槽解码。
`diagnostic_us` 报告这次复制及其完成同步的开销；不再包含原来的标量解码开销。无重试时不复制。
快照上限为4096个 int32；超过上限或 UB 不够保留双份快照时，仍保留次数及耗时统计，
`failed_snapshot_available=false`，失败来源字段留空。新 buffer 布局需两节点重新构建并使用同版本解析器。
最后失败快照描述的是一次采样，不能推断真实的网络到达先后顺序。
来源位图覆盖0～63，超出时 `source_mask_overflow=true`，来源列表仍保留完整快照中解出的 source。
上述累计子耗时已经包含在父区间中，不能再与父区间相加。

Relay 的主机容量按完整512 B包计算，包含每包的末尾标记，并按有效 payload 计算批次数。
BS32/K8 的256个 top-k 条目需要3个包，UB 配置为512个槽（有效容量508），
避免原来256槽仅有254个有效位置而产生“254＋2”两批回退；同时满足单批 route 快路径的输出缓冲要求。

跨机 Combine 的推进开销在 `/COMBINE_PROGRESS` 子轨道单独记录：

| 事件 | 测量范围 |
| --- | --- |
| `AIV_COMBINE_INIT` | 首次构建私有 pending masks 的元数据 DMA 和标量处理；每次算子只记录一次 |
| `AIV_COMBINE_PROGRESS` | 初始化后的单次扫描和提交；记录进入/退出时待发的行数、POST 数、扫描槽位数及扫描/控制累计周期 |
| `AIV_HOST_COMBINE_BARRIER_WAIT` | 已有16卡机内完成屏障的完整调用，包含发布、轮询及本卡同步 |
| `AIV_COMBINE_DRAIN` | 屏障后最终 drain 的父区间，包含子 progress 和退避；记录进入/退出时待发行数 |
| `AIV_COMBINE_RUN` | `/COMBINE_RUN` 轨道记录每个连续段的起点、行数、发送或延后决策、终止原因以及是否曾延后 |

`scan_control_us` 包含游标扫描、ready flag 检查和私有 pending 状态维护，排除初始化、
`RocePutBytes` 调用和 drain 退避；它仍包含测量边界的开销，不能当作纯 GM 访存耗时。
progress/drain 的父区间不能与内部 RDMA 或子 progress 相加。初始化是独立区间，
已全部发送的 progress 快返不产生扫描记录；零待发的最终 drain 仍记录0行，便于确认完成。
这些观测不新增流水排空、跨核同步或通信，不以计时结果决定调度。

解析另输出 `combine_progress_breakdown.csv`，始终保留上述数据，即使 JSON 隐藏 WAIT。
其中 `pending_before` / `pending_after` 是行数；`posts` / `scanned_rows` 只属于 progress，
drain 行的这些字段留空。通过它比较单次推进耗时与最终剩余行数，判断缩减预算是否把工作移到了尾部。
run 的 `stop_reason` 区分 `tensor_end`、`route_gap`（无路由或已发送）、`producer_not_ready` 和 `scan_budget`；
其区间嵌套在 progress 中，并可能包含 RDMA POST，不能重复累计。
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
打包使用 AIV0 自己的 epoch 计数槽，元数据发送前等待 `LocalRailPacked`；payload 先行提交，
与 AIV0 剩余的元数据打包重叠。

跨机输入 payload 在本卡全部 token/scale 准备完成后，按实际 B 一次 Put 到配对 gateway，
不再扫描路由并拆成多个连续 token 段；B 小于预分配容量时只发送实际 B 行。
接收区保留原 token 编号，专家仍按路由元数据读取，多传的行不会增加计算。
该策略用于上述 B≤32 的 A8W4/A8W8 路径，包括没有远端路由的批次。
元数据和 payload 先在同一 QP 提交，owner 完成 reset/shared-input 工作后
发布共用的 `NetPayloadCommit`。HNS1825 使用已确认的网卡合同：每个 WQE 的末分片自动具有 SO，
覆盖同一 QP 的前序 WQE；远端观察到该分片时，前序数据已在 GM 可见。因此独立的4 B commit
WQE 可直接跟在数据之后，Dispatch 不执行显式 quiet；SHMEM 无需设置 WQE 的 SO 字段。
接收 leader 获取这个标记后立即发布 `RelayPayloadReady` 和 `MetaReady`，然后转发 metadata；
接收就绪不等待本卡反方向的 CQ 完成。接收侧仍保留 DCache 失效和 scalar→MTE2 acquire。
公共输入 barrier 保留，统一 commit 和接收就绪仍须在它之前发布。
HNS1825 的 Dispatch、Combine 发送源以及完整32位 `Sequence` 在本轮保持不变；
Unpermute 只读独立的接收区。输出就绪屏障之后，通信 owner 统一 quiet 一次并发布 `TxDone`，
其他 AIV 同时执行 Unpermute；kernel 完成等待两部分工作全部结束。
下一轮沿用同一 runtime stream，不能在这个完成边界前覆盖源区或 epoch。
XSCALE 没有使用上述 SO 合同，仍保留每阶段 `quiet → commit → quiet`；该优化不依赖 PI-on-chip 开关。
这会放弃 metadata 提前到达时的 relay 重叠机会，整体收益以关闭打点的 perf 为准。
H=3584、B=32 时 payload 为118784 B；相较已测 EP32/K8 balanced 样本的8次7424 B提交，
该策略以每 rank 额外59392 B流量换取一次提交和更少的发送端扫描工作。
性能验证应同时检查 payload POST 次数、quiet、输入准备结束及首 GMM 时间，
并以关闭打点的 `rank-max-mean` 判断整体收益，不能仅由 POST 次数推断加速比例。

跨机输入的 gateway 依赖按 AIV1 block 分工：各核从同一份有效 count 表得到本 rank
实际需要的来源，并在原始 count 被前缀和覆盖前保存依赖 mask。数量表及前缀写回完成后立即
发布 count-ready，允许 AIC/AIV0 提前准备 GMM 状态和权重。随后唯一检查者确认对应 gateway
就绪，向本卡独立的512 B槽发布当前 epoch；AIV1 等到所需本卡槽就绪后才读取 payload。
各检查者先完成自己的检查和发布，
再消费本卡通知，避免把并行检查串行化。28个 AIV1 都需要16个来源的用例中，
gateway 检查调用由448次降为16次；这些是依赖检查调用数，未就绪时仍会轮询。
新增16×512 B槽及对齐 padding 纳入现有输入屏障前的 workspace reset，
不改变 peermem 通信布局，也不新增全核屏障。转发前显式执行
[`DataSyncBarrier<DDR>`](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/sync_control/intra_core_sync/DataSyncBarrier_ISASI.md)，
消费者保留 scalar→MTE2 顺序保障。A8W4/A8W8 共用此协议。

A8W4 的跨机 Combine 在 AIV1 的计算间隙分批推进：仍有本核路由或共享 activation 时，每次最多提交2个 Put、
扫描64个 `[token, top-k]` 槽位；这些前台工作完成后扩大为最多8个 Put和一个完整槽位扫描。
输出 export 前的最后一次推进也使用扩大后的预算。扫描游标跨调用保留，已发送槽位不再参与就绪判定；
全部发送后立即返回。固定物理 AIV1 保留 activation、dispatch 和远端推进，
本地 Combine 搬运交给其余 AIV1：28个 AIC 时由剩余27个 AIV1 均衡分配完整行，
在27个 consumer 的空间内按 Combine wave 和全局行前缀轮换，确保每行只搬运一次。
GMM2 仍使用原28个生产核及其实际完成计数，不能用27个搬运核代替生产核数。
owner 对每个 Combine slice 保留非阻塞生产完成检查，就绪后推进 slice 并尝试远端发送，
再回到循环入口检查 activation；尚未就绪时继续其他任务和有预算的远端推进。
这避免 owner 因无本地行而立即跳完所有 slice、提前进入 SyncAll。非 RoCE、A8W8 路径
及仅一个 AIC 的回退路径沿用原本地 Combine 分工；QP1 始终由固定物理 AIV1 独占。
最终 host barrier 后完整 drain，HNS1825 直接提交 Combine commit，再等待对端 commit、执行 acquire
和全 AIV 同步。随后 owner 执行 quiet，其他 AIV 同时 Unpermute。主机配置和设备分工统一排除物理 owner，
例如56个 AIV 时由55个消费核完整分配输出行，owner 不再承担输出 token；不新增同步屏障。
两端都必须先提交再等待；没有远端行也不能省略
Dispatch/Combine commit。host barrier 关闭本机的生产者和旧 Dispatch 读取，双向 commit 继续保护跨轮复用。
单次 Put 内部仍可能因队列回压等待，这些计数预算并非硬性时间上限。
扫描在单次调用内缓存当前 token 的 pending mask，通过递增槽位推进，避免逐行除模；
连续发送段在 Put 返回后按 token 清除对应位。扫描仍逐槽计数，保留洞、游标绕回、
预算截断和相邻行合并。少于4行且前后仍有待发行的短段，允许同一起点最多延后一次，
继续扫描其他位置并返回计算；再次遇到该起点就允许发送，不等待邻行的生产者。
没有相邻待发行的孤立段立即发送。最终 drain 关闭延后和预算限制，立即提交所有已就绪段。
这些规则不以 timer 结果决定调度。完成当次扫描或达到 Put 预算后返回，
后续调用在全部行已提交时直接返回。提交完成不代表传输完成，后者仍由后续 quiet 保证。

HNS1825 每轮显式 quiet 从4次减为1次，包含在 kernel 总计时内；不能将 quiet 的耗时简单乘以
删除次数当作端到端收益。SDK 的队列回压回收仍保留，每个 WQE 继续产生 CQE。
每轮 WQE 数最多为 `B * topK + 4`，是否触发内部回收取决于实际 SQ/CQ 深度和预留空间。
网络 commit、Dispatch 本地就绪、host 握手及最终 Combine 行等待加入60秒停滞超时诊断；只有 QP owner
能在等待期间稀疏检查已 ready 的 CQE，不等待尚未完成的 CQE，不允许接收 leader 并发消费 CQ。
正常短等待不增加 CQ DMA，最终 quiet 负责统一完成检查和队列回收。硬件 `SyncAll` 本身没有新增
可取消机制，软件等待超时会打印上下文并 trap；日志 `kind=0/1/2/3` 分别表示 commit、
Dispatch 本地就绪、host 生产者屏障、Combine 行，另含 peer、expected、observed。设备故障后的清理由 runtime 负责。

ready 探测通过一次跨步 DMA，将最多8个512 B flag 槽的首32 B收集到 owner 专用256 B UB。
读取沿用默认 L2 缓存策略 `CACHE_MODE_NORMAL`，DMA 完成后仅将匹配当前完整 sequence 的结果保留为标量位图，
并完成 scalar→MTE2 顺序保障，再允许 Put 复用 scratch。普通 Put 的后端在返回前完成其 UB 搬运，
因此下一批快照可以复用同一空间；这里不使用需要保留 workspace 的 aggregate Put。
预取可以包含洞和已发送槽位，但实际发送仍受 pending mask 约束；每批不越过物理行末尾或剩余扫描预算。
位图只在当前 progress 调用内有效，晚于快照发布的行在后续扫描重新读取，不缓存跨调用的未就绪结果。
静态全待发场景下64行由8次 DMA读取，逻辑扫描计数仍为64；这不是设备耗时的八倍加速承诺。

Combine 路由首次初始化使用 owner 专用256 B UB 分块 DMA 读取已接收的 rail 元数据；
BS32、K8 时为5次256 B DMA，跳过每个512 B packet 的末尾 marker。
该初始化优化也适用于 A8W8，其 Combine 使用最多8个 Put、一次完整扫描和相同的短段合并规则。
更新后两台均需重新构建安装 timer OFF 版本；按上述 BS32 流程先执行 `run` 校验精度，
再执行一次 `perf`（默认20次 warmup、100次采样，统计后50次），将两个指标与前一版本比较。
本轮 ready 批量探测的比较基线为 owner 搬运卸载版本的241.830/281.499 μs
（rank-min mean / rank-max mean）；全体 rank、全部选中轮次的均值为262.111 μs，可作补充观察。
常规迭代不自动运行 A/B 或重复确认；需要定位阶段时，再单独采集
`trace --timer-ranks 0,14,16,30 --include-wait`，检查输入 payload、计算尾部、
Combine POST 分布和 quiet/commit 等待。CPU 回归不能替代设备精度与性能检查。

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
| `--perf-method legacy\|shmem` | 默认 legacy：每批 Exp20；shmem：每轮清 L2、device sleep和SHMEM同步，同样作用于timer采集 |
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

### 融合 GMM1 → GMM2 的独立性能测试（单场景与 EP/BS 扫参）

[`scripts/bench_gmm_chain.py`](scripts/bench_gmm_chain.py) 在单张 Ascend950 上测试：
`npu_grouped_matmul_swiglu_quant_v2`（GMM1 + SwiGLU + MX 量化）→ `npu_grouped_matmul`（GMM2）。
GMM2 直接消费融合算子产出的 FP8 激活及 E8M0 scale；权重为 MXFP4、NZ 格式，最终输出 BF16。
接口按官方 [融合算子 MxA8W4 示例](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul_swiglu_quant_v2.md)
及 [GMM mx 伪量化示例](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md) 接入。
需要支持这些接口和数据类型的 CANN、PyTorch/torch_npu 环境；缺失时直接报错，结果中保留实际版本和算子 schema。

默认复现 EP32、每个源 rank BS32、H3584、I1280、E384、TopK8、S0、balanced 路由，seed=20260922。
复用现有 benchmark 的输入、路由、原始权重和 MXFP8 输入量化，按专家分组重建 Dispatch 的逻辑输出。
每卡12个专家，**实际 GMM 总行数 M=256**，各专家行数为 `[22,22,22,22,21,21,21,21,21,21,21,21]`。
GMM1 的 K/N 为3584/2560，GMM2 为1280/3584。`--bs` 是每个源 rank 的 token 数，不能直接当作 GMM 的 M。

在仓库根目录、已激活的 CANN/torch_npu 环境中直接运行，无需重编 DeepEP、初始化 SHMEM 或启动 torchrun：

```bash
# CPU 预览实际路由和矩阵形状，不导入 Torch，不生成权重。
python experiments/megamoe/scripts/bench_gmm_chain.py --dry-run

# 单卡测量，默认20轮预热、100轮正式采样。
python experiments/megamoe/scripts/bench_gmm_chain.py --device 0

# 可选：NPUGraph replay，减少 Python 下发造成的空隙；结果单独标记 graph。
python experiments/megamoe/scripts/bench_gmm_chain.py --device 0 --mode graph

# 复现逻辑 rank30 的输入和权重，仍只需一张物理卡。
python experiments/megamoe/scripts/bench_gmm_chain.py --device 0 --rank 30 --iters 200

# 一次跑完 EP16/32/64 × BS1/8/16/32/256，共15个场景。
python experiments/megamoe/scripts/bench_gmm_chain.py \
  --device 0 --ep-sizes 16,32,64 --bs-values 1,8,16,32,256 \
  --warmup 20 --iters 100

# 同一批场景使用 NPUGraph replay；也可加 --dry-run 只预览形状。
python experiments/megamoe/scripts/bench_gmm_chain.py \
  --device 0 --ep-sizes 16,32,64 --bs-values 1,8,16,32,256 --mode graph
```

`--device` 选择进程可见的 NPU，`--rank` 选择逻辑专家归属，两者互不绑定。
可通过 `--ep-size/--bs/--hidden/--intermediate/--experts/--topk/--routing/--seed` 调整场景。
此脚本只测路由专家，S固定为0。测试期间应让所选设备空闲。

`--ep-sizes` 和 `--bs-values` 执行笛卡尔积，也可以只传其中一项，例如
`--ep-size 16 --bs-values 1,8,16,32,256`。复数参数与对应单数参数互斥；所有场景在启动前验证专家整除和逻辑 rank 范围。
未传复数参数时，保持原来的 EP32/BS32 单场景行为。
默认 balanced 路由下，各逻辑 rank 的专家行数分布相同；默认 `--rank 0` 即复现这种计算形状。
每卡专家数分别为 `384/EP = 24/12/6`，GMM 总行数为 `BS × TopK`：

| 每个源 rank 的 BS | 1 | 8 | 16 | 32 | 256 |
| --- | --- | --- | --- | --- | --- |
| 每卡 GMM 总行数 M（TopK8） | 8 | 64 | 128 | 256 | 2048 |

每个专家有 `floor(M / 本卡专家数)` 或多1行。BS1/EP16 有16个空专家，BS1/EP32 有4个空专家；
group list 保留这些0，不删除专家或扩充 token 数。

批量测试在同一张卡上顺序启动独立子进程，每个场景结束后释放设备、graph和profiler资源。
父目录为 `benchmark-runs/gmm-chain/sweep_<随机编号>/`，每个场景的结果放在
`ep<EP>_bs<BS>/rank_<r>_<随机编号>/`，同级 `console.log` 保留该场景完整日志。
顶层 `summary.txt` 提供15场景对比表；`summary.csv` 提供全部5项计时的 mean/median/min/max，
以及EP、BS、M、专家数、精度状态、采样轮次、结果路径；`summary.json` 另保存逐专家行数、命令和失败原因。
每完成一个场景即更新汇总。单场景失败会保留日志、继续后续场景，最后返回非零退出码；
失败或未完成场景的计时留空，不参与性能比较。每次扫参都会创建新目录，避免读到上次的结果。

单场景每次创建独立的 `benchmark-runs/gmm-chain/rank_<r>_<随机编号>/`，可通过 `--output` 更改父目录。
其中 `results.json` 保存场景、精度报告、输入/权重摘要、软件版本、全部采样及汇总；
`summary.txt` 是可读摘要，`profiler/` 保留原始 trace 和 `kernel_details*.csv`。
按现有性能测试口径取正式采样的后半段（奇数向上取整），每项报告 mean/median/min/max，单位均为微秒：

| 指标 | 范围 |
| --- | --- |
| `fused_gmm1_us` | 融合 GMM1 + SwiGLU + MX 量化的 kernel Duration |
| `gmm2_us` | GMM2 的 kernel Duration |
| `kernel_sum_us` | 同一轮两个 kernel Duration 之和 |
| `chain_span_us` | 从融合 kernel 开始到 GMM2 kernel 结束，包含两者间隙 |
| `inter_kernel_gap_us` | 两个 kernel 之间的空隙，eager 模式下可能受 CPU 下发影响 |

两个算子之间及正式轮次之间不插入 synchronize、Event、CPU 拷贝或额外量化。
统计要求每轮恰好两个匹配的 kernel、处于同一设备同一 stream；发现额外 kernel、重复导出、缺失样本或时序异常会报错，保留原始数据供排查。
时间差使用 CSV 十进制原值计算，避免大绝对时间戳转为 `float` 后丢失亚微秒精度。
采样中的 `start_us_exact` / `duration_us_exact` 字符串保留精确值；`inter_kernel_gap_raw_us` 保留有符号间隙。
重叠容差仍为0.01微秒，超过时报告具体间隙、设备/stream及两条记录的 CSV 行号；容差内的负间隙仅在 `inter_kernel_gap_us` 中记为0。
若所用版本把一个算子展开为多个 kernel，需要根据该版本的 trace 调整归属解析后再统计。

默认在计时前按每个非空专家抽取最多2行，与现有 NumPy `expert_forward` 比较；
使用现有混合误差阈值并报告相对 RMSE。该参考包含 BF16 中间舍入，不是融合算子逐位仿真，
不通过时停止性能采样。`--check-rows-per-expert` 可提高覆盖，设为0显式跳过数值参考（仍检查输出形状、类型和有限值），
跳过结果会标为 `explicitly_skipped`，不能视作精度已验证。Graph 模式还会在计时前检查 replay 与 eager 输出一致。

计时排除 Dispatch、Combine、路由权重乘法、输入量化、权重转换和 CPU 校验。
这里得到的是单卡计算链基线，不是 EP16/32/64 张卡同时运行的统计；与已有 Dispatch/Combine 数据相加可作分段估算，
实际端到端 MoE 耗时仍需用一致的数据、执行模式和计时范围另行测量。

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
