---
aliases:
  - "2606.19348v1"
  - "21"
  - "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
  - "DeepSeek-V4"
created: 2026-11-15
---
# DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence

> 技术报告（preview 版），DeepSeek-AI，2026 年 6 月。两个 MoE 模型：**DeepSeek-V4-Pro**（1.6T 总参 / 49B 激活）与 **DeepSeek-V4-Flash**（284B 总参 / 13B 激活），两者都原生支持 **1M token** 上下文。核心卖点不是"更大"，而是**在 1M 上下文下把单 token 推理 FLOPs 降到 DeepSeek-V3.2 的 27%、KV cache 降到 10%**。

## 1. Problem & Motivation

### 1.1 论文针对的具体问题

- **test-time scaling 被 attention 的二次复杂度卡死。** 推理模型靠"想得更久"涨点，但上下文一长，vanilla attention 的算力与显存开销呈平方增长，长思维链与超长上下文的成本不可接受 [§1](2606.19348.pdf#section=1)。
- **长程任务（long-horizon）需要"真的能用"的 1M 上下文。** 复杂 agentic workflow、跨文档分析等场景要求模型在一次前向中看到海量材料，而不仅是"标称支持"。
- **开源模型的能力进步没有解决这个架构级低效。** V3、Qwen3、MiniMax-M2、Kimi-K2 等提升了通用能力，但处理超长序列的底层低效依旧限制了 test-time scaling 的继续扩展。

### 1.2 前人方案的根本瓶颈

- DeepSeek Sparse Attention（DSA，V3.2 已采用）只做了**"选哪些 token 算"**，KV 仍是**逐 token 存储**；1M 上下文时 KV cache 与 attention FLOPs 依然是主瓶颈。
- 纯线性注意力 / 固定压缩方案能省算力，但会牺牲精确检索能力，长上下文"大海捞针"类任务掉点明显。
- Hyper-Connections（HC）通过加宽 residual stream 提升表达力，但多层堆叠时**数值不稳定**，难以规模化。

### 1.3 本文的核心贡献（一句话版）

把"**KV 压缩**（CSA/HCA）+ **稀疏选择**（DSA indexer）+ **流形约束的残差**（mHC）+ **Muon 优化器**"组合起来，在保住 dense 级检索精度的同时，把 1M 上下文的推理成本压到可常态化部署的水平，并用一整套 infra（EP mega-kernel、异构 KV cache、FP4 QAT）把纸面效率落成工程效率。

## 2. Core Contribution：方法拆解

### 2.0 Overview（先建立全局图景）

V4 仍然是 **MoE Transformer + MTP** 的骨架：FFN 用 DeepSeekMoE（1 个 shared expert + 256/384 个 routed experts，每 token 激活 6 个），MTP 深度为 1，配置与 V3 一致 [§2.1](2606.19348.pdf#section=2.1)。真正的改动集中在三处：

1. **Attention 层**：不再每层都做全量 attention，而是把层分成 **CSA（Compressed Sparse Attention）** 与 **HCA（Heavily Compressed Attention）** 两类交替堆叠，并额外挂一条 **sliding window** 分支补局部细节。CSA 先在序列维把每 $m$ 个 token 压成 1 个 KV entry，再用 lightning indexer 选 top-$k$ 个压缩块做 core attention；HCA 压得更狠（$m' \gg m$）但保留 dense attention。二者都把"KV 长度"从 $n$ 降到 $n/m$ 或 $n/m'$，这是 1M 上下文可行的根本原因 [§2.3](2606.19348.pdf#section=2.3)。
2. **残差连接**：用 **mHC** 把 residual stream 加宽 $n_{\text{hc}}$ 倍，并把残差映射矩阵约束到双随机矩阵流形（Birkhoff polytope）上，使映射非扩张（spectral norm $\le 1$），从而在深层堆叠时保持数值稳定 [§2.2](2606.19348.pdf#section=2.2)。
3. **优化器**：大部分参数改用 **Muon**（矩阵级正交化更新），收敛更快、训练更稳；embedding、prediction head、mHC 的静态参数与 RMSNorm 仍用 AdamW [§2.4](2606.19348.pdf#section=2.4)。

围绕这三点，论文还给了大量系统工程（第 3 章）与后训练改造（第 5 章），使"1M 上下文"从架构假设变成可训练、可服务的能力。

### 2.1 继承 V3 的部分与小幅修改

- **MoE 路由**：affinity score 的激活函数从 $\operatorname{Sigmoid}(\cdot)$ 改为 $\operatorname{Sqrt}(\operatorname{Softplus}(\cdot))$；继续用 aux-loss-free 负载均衡，另加一个很小的 sequence-wise balance loss（权重 $10^{-4}$）防止单条序列内极端不均。
- **取消 routing target node 数量约束**，重设并行策略以维持训练效率。
- **前几个 block 的 dense FFN 换成 Hash routing MoE**：按 token ID 的哈希函数决定专家，前 3 个 MoE 层使用；这消除了早期层的路由抖动与负载不均。
- **MTP**：完全沿用 V3，未做修改。

### 2.2 mHC：把残差映射关进 Birkhoff 多面体

**标准 HC 的写法。** 残差流宽度扩为 $n_{\text{hc}} \times d$，记第 $l$ 层前的残差状态为 $X_l \in \mathbb{R}^{n_{\text{hc}} \times d}$，则

$$X_{l+1} = B_l X_l + C_l \mathcal{F}_l(A_l X_l),$$

其中 $A_l \in \mathbb{R}^{1 \times n_{\text{hc}}}$ 是输入映射、$B_l \in \mathbb{R}^{n_{\text{hc}} \times n_{\text{hc}}}$ 是残差变换、$C_l \in \mathbb{R}^{n_{\text{hc}} \times 1}$ 是输出映射，$\mathcal{F}_l$ 是第 $l$ 层（如 MoE）。因为 $A_l X_l$ 仍是 $d$ 维，加宽残差**不影响层内结构**，$n_{\text{hc}}$ 成了一个廉价的新 scaling 轴。但作者发现：多层堆叠时 HC 训练**频繁数值不稳定**。

**mHC 的约束。** 核心是把 $B_l$ 约束到双随机矩阵流形

$$\mathcal{M} \coloneq \left\{ M \in \mathbb{R}^{n \times n} \mid M\mathbf{1}_n = \mathbf{1}_n,\ \mathbf{1}_n^\top M = \mathbf{1}_n^\top,\ M \ge 0 \right\}.$$

这个约束保证 $\|B_l\|_2 \le 1$，残差变换**非扩张**，前向与反传都不会放大信号；且 $\mathcal{M}$ 对矩阵乘法封闭，深层堆叠仍稳定。$A_l, C_l$ 则用 Sigmoid 约束为非负有界（$A_l = \sigma(\tilde A_l)$，$C_l = 2\sigma(\tilde C_l)$），避免信号相互抵消。

**动态参数化 + 投影。** 三个映射的原始参数由"输入相关 + 静态偏置"两部分生成：先把 $X_l$ 展平做 RMSNorm 得 $\hat X_l$，再经低秩投影与可学习门控 $\alpha$ 得到 $\tilde A_l, \tilde B_l, \tilde C_l$。随后 $B_l$ 用 **Sinkhorn-Knopp** 投影到 $\mathcal{M}$：先 $M^{(0)} = \exp(\tilde B_l)$ 保证正性，再交替做行/列归一化

$$M^{(t)} = \mathcal{T}_r\left(\mathcal{T}_c\left(M^{(t-1)}\right)\right),$$

迭代 $t_{\text{max}} = 20$ 次得到 $B_l$。工程上 mHC 的墙钟开销被压到 1F1B pipeline stage 的 **6.7%**（融合 kernel + 选择性重算 + DualPipe 调整）[§3.4.2](2606.19348.pdf#section=3.4.2)。

### 2.3 Hybrid Attention：CSA + HCA

#### CSA（Compressed Sparse Attention）

**第一步，压缩 KV。** 对输入 $H \in \mathbb{R}^{n \times d}$ 算出两组 KV 与压缩权重 $C^a, C^b, Z^a, Z^b \in \mathbb{R}^{n \times c}$，再把每 $m$ 个 KV entry 按权重 + 可学习位置偏置 $B^a, B^b$ 压成一个：

$$\left[S^a_{mi:m(i+1)-1}; S^b_{m(i-1):mi-1}\right] = \operatorname{Softmax}_{\text{row}}\left(\left[Z^a_{mi:m(i+1)-1} + B^a;\ Z^b_{m(i-1):mi-1} + B^b\right]\right),$$

$$C^{\text{Comp}}_i = \sum_{j=mi}^{m(i+1)-1} S^a_j \odot C^a_j + \sum_{j=m(i-1)}^{mi-1} S^b_j \odot C^b_j .$$

注意 softmax 是在 $2m$ 个元素上做的，且 $C^b$ 的窗口与上一个压缩块 $C^a$ 重叠——**重叠压缩**让块边界的信息不被硬切断，最终序列长度压到 $1/m$（$m=4$）。

**第二步，稀疏选择。** 用 **lightning indexer** 给每个 query 打分：query 先低秩压到 $\mathbf{c}^Q_t$，再上投影出 $n^I_h$ 个 indexer head；对第 $s$ 个压缩块

$$I_{t,s} = \sum_{h=1}^{n_h^I} w^I_{t,h} \cdot \operatorname{ReLU}\left(\mathbf{q}^I_{t,h} \cdot K^{\text{IComp}}_s\right),$$

取 top-$k$（Flash $k=512$，Pro $k=1024$）个压缩块进 core attention。indexer 自身也复用同一套压缩流程，键是压缩后的 $K^{\text{IComp}}$。

**第三步，core attention。** 选中的压缩块**同时充当 key 和 value**（shared-KV MQA），query 由同一个 $\mathbf{c}^Q_t$ 上投影得到，因此 KV 只有一份、且被所有 query head 共享——这是 KV cache 能压到极小的关键。输出端再用 **grouped output projection**：把 $n_h$ 个头分成 $g$ 组，每组先降到 $d_g$ 维再拼回 $d$ 维，避免 $c n_h$ 维直接投影的开销（Pro：$n_h=128, c=512, g=16, d_g=1024$）。

#### HCA（Heavily Compressed Attention）

结构同 CSA 的压缩 + MQA + grouped projection，但**压缩率 $m'=128$（远大于 $m=4$）、不做重叠压缩、也不做稀疏选择**——所有压缩块都参与 dense attention。它承担"粗粒度全局视野"，CSA 承担"细粒度 + 稀疏检索"，两者交替堆叠。

#### 其他关键细节（容易被忽略但影响很大）

- **QK RMSNorm**：core attention 前对每个 head 的 query 与压缩 KV 做 RMSNorm，防止 attention logits 爆炸（也因此 V4 不需要 QK-Clip）。
- **Partial RoPE + 反向 RoPE**：只对最后 64 维加 RoPE；由于压缩 KV 同时是 key 和 value，输出会带上绝对位置信息，于是对每个 $\mathbf{o}_{t,i}$ 的最后 64 维再施加位置 $-i$ 的 RoPE，使输出恢复**相对位置**语义。
- **Sliding window 分支**：每个 query 额外保留最近 $n_{\text{win}}=128$ 个未压缩 KV，弥补"query 看不到自己所在压缩块内其他 token"的缺陷，并保证因果性。
- **Attention sink**：分母加可学习 $\exp(z'_h)$，允许每个 head 的总 attention 质量不等于 1（甚至接近 0），提升长上下文稳定性。

#### 效率账（论文最硬的一组数）

- KV 采用**混合精度存储**：RoPE 维用 BF16，其余用 FP8，KV cache 体积相对纯 BF16 接近**减半**；indexer 的 attention 计算用 **FP4**。
- 相比 BF16 GQA8（head dim 128）这一常见配置，V4 在 1M 上下文下 KV cache 只有其 **约 2%**。
- 相比已经很高效的 V3.2：1M 上下文时 **V4-Pro 只用 27% 单 token FLOPs、10% KV cache**；**V4-Flash 只用 10% FLOPs、7% KV cache** [§1](2606.19348.pdf#section=1)。

### 2.4 Muon 优化器

除 embedding、prediction head、mHC 静态参数/门控、RMSNorm 外，其余模块全用 Muon：

$$M_t = \mu M_{t-1} + G_t,\qquad O'_t = \operatorname{HybridNewtonSchulz}(\mu M_t + G_t),$$
$$O_t = O'_t \cdot \sqrt{\max(n,m)} \cdot \gamma,\qquad W_t = W_{t-1}(1 - \eta\lambda) - \eta O_t .$$

正交化用**两段式 Newton-Schulz**：前 8 步用 $(a,b,c) = (3.4445, -4.7750, 2.0315)$ 快速把奇异值推向 1，最后 2 步换成 $(2, -1.5, 0.5)$ 精确稳定在 1。$\gamma$ 把更新 RMS 缩放到 0.18，从而复用 AdamW 的学习率超参。由于 attention 里有 QK RMSNorm，V4 **不使用 QK-Clip**。

### 2.5 基础设施（论文花了整整一章，说明其分量）

- **细粒度 EP + 单 kernel 融合**：把 MoE 层的 Dispatch / Linear-1 / Linear-2 / Combine 拆成按 expert **wave** 调度的流水线，通信藏在计算下面。相对强 baseline 通用推理 **1.50–1.73×**，RL rollout 等延迟敏感场景最高 **1.96×**；已开源为 DeepGEMM 的 **MegaMoE**。论文还给出硬件设计建议：当 $C/B \le 2d = 6144$ FLOPs/Byte 时通信可被完全隐藏，盲目堆带宽收益递减 [§3.1](2606.19348.pdf#section=3.1)。
- **TileLang**：用 DSL 写 fused kernel；Host Codegen 把 host 侧校验从几十~几百 µs 降到 **<1 µs**；集成 Z3 做整数表达式形式化分析（QF_NIA），编译开销仅几秒；默认关闭 fast-math，支持 bitwise 可复现。
- **Batch-invariant / deterministic kernel**：双 kernel 解码策略（整条序列单 SM 算 + 尾 wave 多 SM 算但保持相同累加顺序）解决 wave quantization；弃用 cuBLAS 与 split-k；attention backward 用 per-SM buffer + 全局确定性求和；MoE backward 用 token 顺序预处理 + buffer 隔离。
- **训练框架**：Muon 与 ZeRO 冲突 → hybrid ZeRO（背包分配、padding <10%、DP 超出时冗余计算）；MoE 梯度 BF16 随机舍入 + all-to-all 后 FP32 本地求和；压缩注意力的 **two-stage contextual parallelism**；基于 TorchFX 的**张量级 activation checkpointing**。
- **推理框架**：异构 KV cache（classical cache 按 $\mathrm{lcm}(m,m')$ 分块 + state cache 存 SWA 与未满压缩块的尾部 token）；on-disk KV cache 支持 shared-prefix 复用，SWA 部分给出 full / periodic checkpoint / zero 三种策略，按部署场景取舍。

### 2.6 预训练

- 数据：32T+ token，在 V3 语料上强化长文档（论文、技术报告）、多语言长尾与 agentic 数据；沿用 tokenizer（128K 词表）与 FIM；新增 **sample-level attention masking**。
- 规模：Flash 32T token（batch 75.5M，peak LR $2.7\times10^{-4}$），Pro 33T token（batch 94.4M，peak LR $2.0\times10^{-4}$）；序列长度 4K → 16K → 64K → 1M 逐步扩展。
- 稀疏注意力课程：先用 dense attention 预热 1T token，在 64K 阶段引入稀疏，并先单独 warmup lightning indexer。
- **训练稳定性**（论文明确说"机理尚不完全清楚，但有效"）：
  - **Anticipatory Routing**：用历史参数 $\theta_{t-\Delta t}$ 算路由、当前参数 $\theta_t$ 算特征，提前一步取数据并缓存路由索引；仅在检测到 loss spike 时触发短回滚 + 该模式，额外墙钟开销约 20%，整体可忽略。
  - **SwiGLU clamping**：线性分量截断到 $[-10, 10]$，gate 上界截到 10。

### 2.7 后训练

- **两阶段范式**：先按数学、代码、agent、指令跟随等领域各自 SFT + GRPO 训出专家，再用 **multi-teacher On-Policy Distillation（OPD）** 合并成一个统一模型（10+ 个教师）。
- **三种推理档位**：Non-think / Think High / Think Max，用 `<think>` 标签区分；Max 额外注入一段"绝对最大努力"的系统提示（[§5.1.1](2606.19348.pdf#section=5.1.1)）。评测时三档上下文窗口分别为 8K / 128K / 384K。
- **Generative Reward Model（GRM）**：难验证任务不再训标量 reward model，而是让 actor 自身充当 judge，并对 GRM 直接做 RL，把"评判能力"与"生成能力"联合优化。
- **工具调用**：新 schema 用 `|DSML|` 特殊 token + XML 格式，显著降低转义失败与调用错误。
- **Interleaved thinking**：工具调用场景下**跨用户轮次保留完整推理历史**（V3.2 会在新用户消息时丢弃）；普通对话仍丢弃，以省 token。
- **Quick Instruction**：把"是否搜索 / 生成 query / 判断权威性 / 领域识别"等辅助任务编码成特殊 token 追加在输入后，复用已有 KV cache，省掉额外小模型与重复 prefill，降低 TTFT。
- **OPD 用 full-vocabulary KL**（而非 token-level 近似），梯度方差更小；工程上用"只缓存教师最后一层 hidden state + 按教师排序样本"避免物化 $|V|>100$k 的 logits。
- **FP4 QAT**：MoE 专家权重与 CSA indexer 的 QK 路径用 MXFP4；FP4→FP8 反量化无损（FP8 多 2 个指数位），可完全复用 FP8 训练框架；index score 从 FP32 降到 BF16 使 top-k 选择器 **2× 加速**且保持 **99.7%** 召回。
- **可抢占 rollout**：token 粒度 WAL，抢占/故障后从断点续跑；论文特别指出"从头重生成"会引入**长度偏差**（短回复更易存活），数学上不正确。
- **DSec 沙箱**：Rust 实现（Apiserver / Edge / Watcher），四种执行底座（Function Call / Container / microVM / fullVM），EROFS + overlaybd 分层镜像，单集群支持数十万并发沙箱，轨迹日志支持快进与确定性重放。

## 3. 实验与 SOTA 对比

### 3.1 设置

- Base 评测覆盖世界知识、语言理解与推理、代码与数学、长上下文四个维度，统一内部框架、严格一致设置 [§4.3](2606.19348.pdf#section=4.3)。
- 主评测对比 Opus-4.6-Max、GPT-5.4-xHigh、Gemini-3.1-Pro-High、K2.6-Thinking、GLM-5.1-Thinking；长上下文用 **MRCR 1M** 与 **CorpusQA 1M**，并重跑了 Opus 4.6 / Gemini 3.1 Pro（GPT-5.4 因 API 大量失败未评）。
- Agent 评测：Terminal Bench 2.0、SWE-Verified / Pro / Multilingual、BrowseComp、HLE w/ tools、GDPval-AA、MCPAtlas、Toolathlon；代码 agent 统一 500 步、512K 上下文。

### 3.2 Base 模型：小模型打败大模型

| Benchmark | V3.2-Base（37B act / 671B） | V4-Flash-Base（13B / 284B） | V4-Pro-Base（49B / 1.6T） |
|---|---|---|---|
| MMLU | 87.8 | 88.7 | **90.1** |
| MMLU-Pro | 65.5 | 68.3 | **73.5** |
| Simple-QA verified | 28.3 | 30.1 | **55.2** |
| FACTS Parametric | 27.1 | 33.9 | **62.6** |
| HumanEval | 62.8 | 69.5 | **76.8** |
| MATH | 60.5 | 57.4 | **64.5** |
| LongBench-V2 | 40.2 | 44.7 | **51.5** |

**读法**：V4-Flash 用约 1/3 的激活参数在多数 benchmark 上超过 V3.2-Base，说明"架构 + 数据 + 训练"的收益已经能抵消参数劣势；V4-Pro-Base 则在知识密集型任务上大幅拉开（SimpleQA 28.3 → 55.2，FACTS 27.1 → 62.6）。

### 3.3 主结果：V4-Pro-Max vs 前沿模型

| Benchmark | V4-Pro-Max | Opus-4.6-Max | GPT-5.4-xHigh | Gemini-3.1-Pro-High |
|---|---|---|---|---|
| MMLU-Pro | 87.5 | 89.1 | 87.5 | **91.0** |
| SimpleQA-Verified | 57.9 | 46.2 | 45.3 | **75.6** |
| GPQA Diamond | 90.1 | 91.3 | 93.0 | **94.3** |
| HLE | 37.7 | 40.0 | 39.8 | **44.4** |
| LiveCodeBench | **93.5** | 88.8 | – | 91.7 |
| Codeforces (rating) | **3206** | – | 3168 | 3052 |
| IMOAnswerBench | 89.8 | 75.3 | **91.4** | 81.0 |
| MRCR 1M | 83.5 | **92.9** | – | 76.3 |
| CorpusQA 1M | 62.0 | **71.7** | – | 53.8 |
| Terminal Bench 2.0 | 67.9 | 65.4 | **75.1** | 68.5 |
| SWE Verified | 80.6 | **80.8** | – | 80.6 |
| BrowseComp | 83.4 | 83.7 | 82.7 | **85.9** |

**读法**：
- **开源 SOTA 成立**：SimpleQA-Verified 比所有开源 baseline 高 **20 个绝对百分点**；Codeforces 3206 为全场最高，人类榜约第 23 名；LiveCodeBench 也第一。
- **与闭源前沿仍有 3–6 个月差距**：知识类（MMLU-Pro / GPQA / HLE）落后 Gemini-3.1-Pro，Apex 仅 38.3（Gemini 60.9）。
- **长上下文是相对强项**：MRCR 1M 83.5 超过 Gemini 的 76.3，但落后 Opus 的 92.9；128K 内检索性能稳定，之后开始下降但 1M 处仍可用。
- **agent 是短板**：Terminal Bench 67.9、SWE Pro 55.4，落后 GPT-5.4 / Opus，但 MCPAtlas 73.6、Toolathlon 51.8 说明泛化性不差（不是只对内部框架过拟合）。

### 3.4 Flash 与 Pro 的差距

| Benchmark | Flash-Max | Pro-Max |
|---|---|---|
| SimpleQA-Verified | 34.1 | 57.9 |
| HLE | 34.8 | 37.7 |
| Codeforces | 3052 | 3206 |
| MRCR 1M | 78.7 | 83.5 |
| Terminal Bench 2.0 | 56.9 | 67.9 |

**规律**：知识类差距最大（参数量决定知识容量），推理类在给足 thinking budget 后可接近，agent 类在复杂任务上仍有明显差距。

### 3.5 形式数学与真实场景

- **Putnam-2025**：hybrid formal-informal 流程（自然语言候选解 + 自验证筛选 + Lean 形式化）达到 **120/120** 全对；agentic 设置下（Lean v4.28.0-rc1 + LeanExplore，最多 500 次工具调用）也取得 SOTA。
- **中文写作**：功能写作对 Gemini-3.1-Pro **62.65% vs 34.10%** 胜率；创意写作指令遵循 60.0%、写作质量 77.5%；但在高复杂度/多轮 prompt 上 **45.9% vs 52.0%** 输给 Claude Opus 4.5。
- **搜索**：agentic search 对 RAG **61.7% vs 18.3%** 胜率，成本仅略高（16.2 次工具调用、13.6K prefill vs 10.5K）。
- **白领任务**：30 个中文专业任务、13 个行业，人工盲评对 Opus-4.6-Max **非负率 63%**，优势在任务完成度与内容质量，弱在格式约束遵循与长文压缩。
- **内部代码 agent**：30 个真实研发任务通过率 **67%**（Sonnet 4.5 47%、Opus 4.5 70%、Opus 4.6 Thinking 80%）；85 名内部开发者中 52% 认为可作为默认编码模型，39% 倾向是，<9% 否。

### 3.6 归因分析（证据强度说明）

论文**没有 ablation 表**（技术报告体例），归因主要靠三条间接证据：
1. **同架构不同规模**：Flash 13B 激活即超过 V3.2 37B 激活，说明架构/数据/训练的共同收益大于参数缩减的损失，但无法拆分三者各自贡献。
2. **效率归因**：Fig. 1 右侧直接给出 FLOPs / KV cache 随上下文长度的曲线，CSA/HCA 的压缩率与 top-$k$ 是解析可算的，这部分归因最扎实。
3. **effort scaling**：HLE 与 Terminal Bench 随 reasoning effort 单调上升，且 V4 在 HLE 上比 V3.2 更 token-efficient，说明"更长思考"确实被有效利用。

## 4. 局限与未来方向（论文自述）

- **架构偏复杂**：为了控制风险保留了大量"初步验证过"的组件与 trick，导致结构臃肿；后续要做"principle-driven"的消融与精简，在不掉点的前提下把设计蒸馏到最本质的形态 [§6](2606.19348.pdf#section=6)。
- **训练稳定性机理不清**：Anticipatory Routing 与 SwiGLU Clamping 有效，但为什么有效"仍是开放问题"；作者计划系统研究并加强内部指标监控。
- **稀疏性只做了 MoE + 稀疏注意力**：未来探索新的稀疏维度（如更稀疏的 embedding 模块）。
- **低延迟**：继续研究低延迟架构与系统技术，让长上下文部署与交互更跟手。
- **长程多轮 agent**：明确列为继续迭代方向。
- **多模态**：正在把多模态能力纳入模型。
- **数据**：持续改进数据 curation 与合成策略。

## 5. 疑点与阅读提示

- **"preview version"**：作者自己定性为预览版，部分结论（尤其 agent、写作）是内部评测，外部不可复现；对 K2.6 / GLM-5.1 有空白项（API 繁忙），GPT-5.4 长上下文缺席，跨模型比较并不完全对称。
- **训练算力/硬件未披露**：论文只提到 EP 方案在 NVIDIA GPU 与华为昇腾 NPU 上验证，**训练 GPU 型号、卡数、总 FLOPs 均未明确给出**（Not explicitly specified in text）。
- **效率数字是"估算"**：27% / 10% 是单 token 推理 FLOPs（等效 FP8）与 KV cache 的估算值，不是端到端吞吐或时延；FP4 的潜在 1/3 收益要等未来硬件。
- **配图说明**：本次环境无法把 `source/figures/*.pdf` 栅格化为图片，故正文以图号 + 章节链接指代。关键图可直接打开：[架构总览](source/figures/basic_arch.pdf)、[CSA](source/figures/CSA.pdf)、[HCA](source/figures/HCA.pdf)、[EP 流水线](source/figures/mega_moe_pipeline.pdf)、[KV cache 布局](source/figures/kv_cache.pdf)、[性能与成本](source/figures/dsv4_performance.pdf)、[MRCR](source/figures/mrcr.pdf)。
- **本笔记的 anchor 限制**：该 paper 尚未生成 `layout.json`，段落/图/表级 anchor 无法校验，故统一使用章节级链接（`#section=`）。
