# LLM / Agent Systems Research Engineer：分阶段学习路线与实践手册

> 基于原始文档整理（整理日期：2026-11-26）。已将原附录内容合并至各对应阶段与总览章节，在保留清单条理的同时将适合横向对照与映射的内容改为表格；开始实践前请核对各项目最新 README、Release、安装说明与兼容性要求。

## 目录

* [一、背景与总原则](#一背景与总原则)

* [二、阶段路线总览与最终能力目标](#二阶段路线总览与最终能力目标)

* [三、分阶段学习计划](#三分阶段学习计划)

  * [阶段 0：BalatroEval 真实 Agent 最小闭环](#阶段-0balatroeval-真实-agent-最小闭环)

  * [阶段 1：Python 工程与 PyTorch 基础](#阶段-1python-工程与-pytorch-基础)

  * [阶段 2：Transformer 实现与小规模预训练](#阶段-2transformer-实现与小规模预训练)

  * [阶段 3：Post-training（SFT/LoRA、DPO、PPO/GRPO）](#阶段-3post-trainingsftloradpoppogpro)

  * [阶段 4：Agent 工程与模型评价](#阶段-4agent-工程与模型评价)

  * [阶段 5：推理系统与 vLLM](#阶段-5推理系统与-vllm)

  * [阶段 6：分布式训练、集合通信与 CUDA 进阶](#阶段-6分布式训练集合通信与-cuda-进阶)

* [四、近期行动与资源优先级](#四近期行动与资源优先级)

* [五、通用学习方法](#五通用学习方法)

## 一、背景与总原则

* **个人背景**：具备 C++、复杂工程研发与一定 CUDA/LibTorch 经验；LLM 理论广度较好，但 Python / PyTorch / 训练独立编码不足。当前聚焦 **BalatroEval**，长期目标是打通预训练、Post-training、Agent Evaluation 与推理系统。

* **硬件约束**：**NVIDIA RTX 4060 Laptop**（显存有限）。坚持小模型、短序列、小 batch、梯度累积、LoRA/QLoRA 与可复现实验；近期不以多卡或大规模训练为前提。架构早于 Hopper/Blackwell，WGMMA、TCGen05、Blackwell 专属路径及多 GPU 分布式实验未必能本机直跑，以本机实际 Compute Capability、CUDA Toolkit 与驱动为准。

* **仓库使用原则**：每阶段聚焦一个主线仓库，其余作对照或参考，避免同时装大量依赖或过早跳入复杂 RLHF 框架，切忌同时开十几个仓库。

  * **主线实践**：跑通、改代码、留结果。

  * **源码学习**：读核心模块、理数据流。

  * **进阶参考**：重架构理解，暂不强求本地全量运行。

* **四维验收标准**：

  * **理解**：说清职责、输入输出与数据流。

  * **实现**：独立完成核心功能。

  * **调试**：边界测试、定位异常与根因定位。

  * **迁移**：复用至新任务或自有项目。

* **学习资源分工原则**：Runoob 补语法与工具基础，开源仓库学工程实现，论文理问题与方法，自建项目与实验证掌握——四者不可互相替代。

## 二、阶段路线总览与最终能力目标

* **阶段路线与交付物总览**：

| 阶段 | 建议时间窗口 | 主目标 | 核心交付物 |
| :--- | :--- | :--- | :--- |
| **0. 当前短期冲刺** | 2026-11-26 至 2026-12-05 | BalatroEval 真实 Agent 最小闭环 | 一次完整 Episode、可回放 Trace、一个 Bad Case |
| **1. Python 工程与 PyTorch** | 2026-12 至 2027-01 | 从能读代码到能独立写训练程序 | 可测试的 Python 项目、MLP 训练项目 |
| **2. Transformer 与小规模预训练** | 2027-02 | 独立实现 Decoder-only Transformer | 小 GPT 训练、验证、生成报告 |
| **3. Post-training** | 2027-03 至 2027-04 | SFT/LoRA、DPO，进而理解 PPO/GRPO | 一次微调实验、一份对照评价 |
| **4. Agent / Evaluation 深化** | 与阶段 0–3 并行，2027-04 前后系统整理 | 从 Agent 可运行走向评价实验 | 可复现的 Agent 评价实验与失败归因 |
| **5. 推理系统 / Infra** | 2027-04 至 2027-05 | 部署、测量、解释推理性能 | vLLM 压测报告、性能瓶颈分析 |
| **6. 分布式训练与 CUDA 进阶** | 2027-05 以后持续 | 从单卡实验迁移到系统级理解 | 小规模通信实验、算子性能分析 |

* **节奏约束**：进度按验收推进而非死卡日历；未通过独立实现与调试验收，不盲目进入下一阶段。

* **最终能力与可展示证据清单**：

| 能力 | 可展示证据 |
| :--- | :--- |
| **Python 工程** | 一个有模块边界、测试、日志和配置的项目 |
| **PyTorch** | 独立训练、验证和调试的模型项目 |
| **Transformer** | 从零实现并训练的小型 Decoder-only GPT |
| **Pre-training** | 完整的数据、tokenizer、训练和验证流程 |
| **Post-training** | SFT/LoRA 与 DPO 实验及前后对照 |
| **Agent** | 可运行的工具调用、状态管理和错误恢复流程 |
| **Evaluation** | 可复现任务、确定性验证器、Trace 和失败归因 |
| **Inference** | 小模型部署、TTFT/TPOT/吞吐/显存测量 |
| **AI Infra** | 对一次性能瓶颈进行 profiler 分析和解释 |
| **CUDA** | 一个正确性可验证、性能可测量的小算子 |

* **核心成果线建议**：最终以 **BalatroEval + 小型 GPT 训练/微调项目 + 推理性能分析** 作为三条核心成果线，分别印证 Agent/Evaluation、模型训练与 Infra 能力，避免堆砌互不关联的 demo。

## 三、分阶段学习计划

### 阶段 0：BalatroEval 真实 Agent 最小闭环

* **目标与边界**：

  * 优先验证现有 MCP Server 能否被真实 Agent 调用。

  * 跑通闭环：`Observation → 模型决策 → MCP 工具调用 → 动作执行 → 状态更新 → Episode 终止 → Trace 落盘`。

  * 先用真实 API、固定初始状态和短任务完成单次 Episode，暂不扩充复杂场景或宣称具备完整长期规划能力。

* **核心仓库**：

| 仓库 | 定位 | 建议使用方式 |
| :--- | :--- | :--- |
| [BalatroEval](https://github.com/ChzhanYeb/BalatroEval) | 当前主项目 | 继续在现有代码上推进，不重建框架。优先检查 MCP stdio 通信、工具 schema、Agent host 配置、超时与错误处理。 |
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | 官方 MCP Python SDK | 对照官方 server/client 示例，重点核对 stdio 生命周期、工具注册、参数 schema、异常返回和协议版本。 |
| [MCP Servers](https://github.com/modelcontextprotocol/servers) | 官方参考服务器集合 | 只挑一个简单 server 看目录组织、工具定义和运行方式；不要把所有 server 都部署。 |
| [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) | LLM 评价框架 | 作为后续评价任务、Solver、Tool、Scorer 设计的参考；当前不必迁移 BalatroEval 架构。 |

* **辅助教程映射（Runoob，遇问题查漏补缺）**：

| 教程 | 学习范围 | 使用方式 |
| :--- | :--- | :--- |
| [Python 3](https://www.runoob.com/python3/python3-tutorial.html) | 模块、函数、类、异常、文件、JSON、迭代器、生成器、装饰器、类型注解、虚拟环境 | 查漏补缺；结合 MCP Server 和 Agent loop 修改现有代码 |
| [Git](https://www.runoob.com/git/git-tutorial.html)、[Linux](https://www.runoob.com/linux/linux-tutorial.html) | 分支、提交、日志、Linux 文件/进程/权限、Shell 基础 | 每天在真实项目中使用，不单独刷完所有章节 |
| [FastAPI](https://www.runoob.com/fastapi/fastapi-tutorial.html) | 路由、请求参数、请求体、响应模型、异常处理 | 只有在 BalatroEval 需要 HTTP API 时学习；先做最小 API |

  * 注意：MCP 规范以 [MCP 官方文档](https://modelcontextprotocol.io/) 与 Python SDK 为主，Runoob 仅作语法速查。

* **任务清单**：

  * 独立 Client 或 Host 调用只读工具确认握手与发现；再调有状态动作工具确认参数校验与状态变更。

  * 接入真实模型 API，跑通完整 Episode 并触发终止条件。

  * 落盘可回放 Trace（至少含 `step_id`、`observation`、`available_actions`、`model_response`、`parsed_action`、`tool_call`/`execution_result`、`state_before`/`state_after`、`latency`、`error`）。

  * 人工分析 1 个 Bad Case，区分决策、工具解析与环境执行三类错误。

* **阶段验收清单**：

  * 目标 Agent 确实发现并调用工具（不只是 Server 启动成功）。

  * 工具参数、执行结果与状态变化可从 Trace 核对。

  * 至少一局 Episode 正常终止且 Trace 可还原关键交互。

  * 能对失败案例区分模型决策错误、工具解析错误与环境执行错误。

* **论文与产出清单**：

  * 必读：[ReAct (2022)](https://arxiv.org/abs/2210.03629)、[Toolformer (2023)](https://arxiv.org/abs/2302.04761)。

  * 选读：[AgentBench (2023)](https://arxiv.org/abs/2308.03688)、[τ-bench (2024)](https://arxiv.org/abs/2406.12045)。

  * 产出：绘制 BalatroEval 完整 ReAct/Tool-use 数据流、明确各环节责任归属并列举三类失败样例。

### 阶段 1：Python 工程与 PyTorch 基础

* **目标**：通过小型可运行项目补齐工程化能力与 PyTorch 训练闭环，避免纯刷语法。

* **核心仓库**：

| 仓库 | 主要价值 | 学习重点 | 优先级 |
| :--- | :--- | :--- | :--- |
| [The Hitchhiker's Guide to Python](https://github.com/realpython/python-guide) | Python 工程实践与项目组织 | 模块边界、项目结构、依赖管理、测试与部署习惯 | 高 |
| [uv](https://github.com/astral-sh/uv) | Python 环境和依赖管理 | `pyproject.toml`、虚拟环境、lockfile、可复现安装 | 高 |
| [pytest](https://github.com/pytest-dev/pytest) | 自动化测试 | fixture、参数化测试、异常测试、测试组织 | 高 |
| [PyTorch Tutorials](https://github.com/pytorch/tutorials) | 官方 PyTorch 教程 | Tensor、Autograd、Dataset、DataLoader、训练与保存 | 高 |
| [PyTorch Examples](https://github.com/pytorch/examples) | 短小、可运行的模型示例 | MNIST 等完整训练流程，观察工程代码组织 | 高 |
| [Hugging Face Course](https://github.com/huggingface/course) | Transformers 生态入门 | Tokenizer、Datasets、Transformers、模型加载与微调接口 | 中 |

* **辅助教程与书籍映射**：

| 资源 | 范围 / 主题 | 使用方式 |
| :--- | :--- | :--- |
| Runoob [NumPy](https://www.runoob.com/numpy/numpy-tutorial.html) | ndarray、shape、dtype、索引、广播、矩阵运算、线性代数 | 重点掌握 shape、广播、矩阵乘法；每个知识点写小练习 |
| Runoob [Pandas](https://www.runoob.com/pandas/pandas-tutorial.html) | DataFrame、CSV/JSON、缺失值、过滤、分组、合并、统计 | 用于整理训练数据、实验结果和 Agent Trace |
| Runoob [Matplotlib](https://www.runoob.com/matplotlib/matplotlib-tutorial.html) | 折线图、直方图、散点图、图例、坐标轴 | 绘制训练 Loss、准确率、延迟分布 |
| Runoob [SciPy](https://www.runoob.com/scipy/scipy-tutorial.html) | 常用科学计算和统计模块 | 按需查阅，不需要系统深挖 |
| 《CUDA for LLMs》Ch.4 | 从 PyTorch、反向传播、NumPy、C 到 CUDA 训练 MLP | 完成 MLP 后阅读，建立高层框架与底层实现联系（亦关联阶段 6） |

  * 注意：PyTorch 以 [PyTorch 官方 Tutorials](https://pytorch.org/tutorials/) 为主；`asyncio` 进阶看 [Python asyncio 文档](https://docs.python.org/3/library/asyncio.html)。

* **学习顺序与实践项目清单**：

  * 学习顺序：Python 核心（数据结构/类/异常/迭代器/装饰器/类型注解/模块）→ 工程基础（`uv`/日志/配置/JSON/HTTP/`pytest`/调试/Git）→ 异步基础（结合 BalatroEval 模型调用与 MCP I/O）→ NumPy → PyTorch（Tensor/Autograd/`nn.Module`/Loss/Optimizer/DataLoader）→ 训练循环。

  * 最小实践项目（MLP 分类器，建议 MNIST/Fashion-MNIST）：数据加载与预处理、MLP 定义、训练/验证循环、Loss/Accuracy 记录、Checkpoint 保存与恢复、至少 3 个单测、对比不同学习率或隐藏层宽度。

* **阶段验收清单**：

  * 不依赖 AI 生成核心代码，独立写出结构清晰、可测试、可复现的 PyTorch 训练项目。

  * 能解释一次训练迭代中 Tensor 的 shape、device、dtype 与梯度变化。

* **论文与产出清单**：

  * 必读：[Deep Learning 教材](https://www.deeplearningbook.org/)（前馈网络、反向传播、优化、正则化）、[Adam (2014)](https://arxiv.org/abs/1412.6980)。

  * 选读：[BatchNorm (2015)](https://arxiv.org/abs/1502.03167)、[AdamW (2017)](https://arxiv.org/abs/1711.05101)。

  * 产出：从数学与代码解释单步迭代（前向、Loss、反向、梯度、`Optimizer.step`）及 Adam/AdamW 状态变量与超参数。

### 阶段 2：Transformer 实现与小规模预训练

* **目标**：将 Transformer 原理落地为可运行的小规模预训练与自回归生成实验。

* **核心仓库**：

| 仓库 | 主要价值 | 建议使用方式 |
| :--- | :--- | :--- |
| [build-nanogpt](https://github.com/karpathy/build-nanogpt) | 从空文件逐步构建 GPT 的教学仓库 | **首选**。按 commit 顺序学习模型和训练代码是如何逐步形成的。 |
| [minGPT](https://github.com/karpathy/minGPT) | 简洁、教育性强的 PyTorch GPT 实现 | 阅读模型定义和训练/推理流程；仓库作者已标注其处于半归档状态，适合学习，不作为最新工程模板。 |
| [nanochat](https://github.com/karpathy/nanochat) | 覆盖 tokenizer、预训练、微调、评价和推理的实验框架 | 作为阶段后半段的端到端参考；先读结构，不要一开始照搬全流程。 |
| [LitGPT](https://github.com/Lightning-AI/litgpt) | 多模型预训练、微调和部署 recipe | 学习更完整的训练工程、配置和 LoRA/QLoRA 流程。 |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | 高性能 Attention 的官方实现 | 进阶源码阅读；当前 GPU 不一定适合编译和运行所有优化路径，先做 PyTorch SDPA 对照。 |

* **辅助教程与书籍映射**：

| 资源 | 范围 / 主题 | 使用方式 |
| :--- | :--- | :--- |
| Runoob [NumPy](https://www.runoob.com/numpy/numpy-tutorial.html) | 广播、矩阵乘法、转置、索引、数值类型 | 用 NumPy 手工验证 Attention 的张量形状和简单数值计算；主体实现仍用 PyTorch |
| 《CUDA for LLMs》Ch.8 | FlashAttention、Tiling、Online Softmax | 理解 Attention 后阅读，重点掌握 Tiling 与 Online Softmax，再对照 FlashAttention 实现（亦关联阶段 5、6） |

* **实现、实践与 RTX 4060 约束清单**：

  * 推荐实现顺序：Token Embedding 与位置编码 → Causal Self-Attention → Multi-Head Attention → FFN/SwiGLU → Residual + LayerNorm/RMSNorm → Decoder Block 与 Decoder-only GPT → Causal Language Modeling 数据集与 Next Token Prediction → 训练/验证/Checkpoint/文本生成 → 超参对比。

  * 最小实践项目：字符级或小词表 tokenizer 在小语料训小 GPT，记录训练/验证 Loss、保存配置/随机种子/checkpoint、生成固定 prompt 样例、对比欠拟合/过拟合/训练不稳定表现，并与 PyTorch SDPA 对照输出数值。

  * RTX 4060 约束：小层数、小 hidden size、小 batch + 梯度累积、限制 sequence length；先小规模精度实验，不强求复现 GPT-2 124M 全量训练，`nanochat`/`LitGPT` 完整配置先拆解再跑小配置。

* **阶段验收清单**：

  * 能独立实现 Decoder-only Transformer，完成小规模训练与自回归生成。

  * 能解释数据如何变 token、loss 如何计算、梯度如何更新及过拟合/重复生成原因。

* **论文与产出清单**：

  * 必读：[Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)、[GPT-3 (2020)](https://arxiv.org/abs/2005.14165)、[Chinchilla (2022)](https://arxiv.org/abs/2203.15556)、[LLaMA (2023)](https://arxiv.org/abs/2302.13971)。

  * 架构选读：[RoPE (2021)](https://arxiv.org/abs/2104.09864)、[RMSNorm (2019)](https://arxiv.org/abs/1910.07467)、[GLU Variants (2020)](https://arxiv.org/abs/2002.05202)、[GQA (2023)](https://arxiv.org/abs/2305.13245)、[Switch Transformers (2021)](https://arxiv.org/abs/2101.03961)。

  * 高效 Attention 选读：[FlashAttention (2022)](https://arxiv.org/abs/2205.14135)、[FlashAttention-2 (2023)](https://arxiv.org/abs/2307.08691)。

  * 产出：完成小型 GPT 结构说明、对照实现 Attention/RoPE/RMSNorm/SwiGLU、解释 Chinchilla 数据配比启示与 FlashAttention 优化瓶颈。

### 阶段 3：Post-training（SFT/LoRA、DPO、PPO/GRPO）

* **目标**：循序掌握 SFT/LoRA 与 DPO，再理解 PPO/GRPO，避免一上来陷入复杂 RL 系统。

* **核心仓库**：

| 仓库 | 主要价值 | 建议使用方式 |
| :--- | :--- | :--- |
| [torchtune](https://github.com/meta-pytorch/torchtune) | PyTorch 原生 LLM 后训练 recipe | 阅读单卡 LoRA、SFT、DPO recipe；理解配置、数据管线、训练循环。 |
| [TRL](https://github.com/huggingface/trl) | SFT、DPO、GRPO 等训练器 | **首选工具库**。从 SFTTrainer 开始，再看 DPOTrainer，最后学习 GRPOTrainer。 |
| [LitGPT](https://github.com/Lightning-AI/litgpt) | 预训练、微调、LoRA/QLoRA 的完整流程 | 作为训练脚本和配置管理的对照实现。 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | PPO、GRPO 等 RLHF / Agentic RL 系统 | **进阶阅读**。先理解算法和数据流，暂不要求在笔记本上运行完整分布式训练。 |

* **辅助教程与书籍映射**：

| 资源 | 范围 / 主题 | 使用方式 |
| :--- | :--- | :--- |
| Runoob [Pandas](https://www.runoob.com/pandas/pandas-tutorial.html)、[Matplotlib](https://www.runoob.com/matplotlib/matplotlib-tutorial.html) | 数据清洗、统计、实验曲线和对照图 | 分析 SFT/DPO 前后结果；不需要为了微调去学习完整数据库开发 |
| 《CUDA for LLMs》Ch.9 | 量化、校准、AWQ/NF4 等 | 与 QLoRA、推理量化实验结合（亦关联阶段 5） |

* **推进路径清单**：

  * **SFT / LoRA**：理解 instruction 数据格式与 prompt/completion/labels/loss mask；跑通小模型 SFT；对比全参微调与 LoRA 的参量、显存和速度；用固定测试集比较微调前后行为。

  * **DPO**：理解 chosen/rejected 偏好数据与 policy/reference/log prob/preference loss；用小规模数据完成可复现实验；检查偏好胜率与通用能力变化。

  * **PPO / GRPO**：掌握 policy gradient、advantage、reward、KL penalty；读 TRL `GRPOTrainer` 与 OpenRLHF 架构；先在小型可验证任务做简化实验，不直接接复杂游戏环境做大规模 RL。

* **阶段验收清单**：

  * 至少完成一次 SFT/LoRA 微调，能解释数据处理、loss mask、训练配置与评价方式。

  * 完成一次 DPO 小实验或严格复现实验。

  * 能从算法层面解释 PPO/GRPO 的训练信号与主要稳定性问题。

* **论文与产出清单**：

  * 必读：[InstructGPT (2022)](https://arxiv.org/abs/2203.02155)、[LoRA (2021)](https://arxiv.org/abs/2106.09685)、[QLoRA (2023)](https://arxiv.org/abs/2305.14314)、[DPO (2023)](https://arxiv.org/abs/2305.18290)。

  * 选读：[PPO (2017)](https://arxiv.org/abs/1707.06347)、[DeepSeekMath / GRPO (2024)](https://arxiv.org/abs/2402.03300)、[Alignment Handbook](https://github.com/huggingface/alignment-handbook)（开源训练配方参考）。

  * 产出：绘制 SFT、Reward Model、PPO、DPO、GRPO 数据流与目标对比图，完成 SFT/LoRA 实验及固定测试集前后对照。

### 阶段 4：Agent 工程与模型评价

* **目标**：围绕 **BalatroEval** 从"能调工具"进阶到"连续交互表现评估、失败归因与评价可信度"，不另起重复框架。

* **核心仓库**：

| 仓库 | 主要价值 | 建议使用方式 |
| :--- | :--- | :--- |
| [BalatroEval](https://github.com/ChzhanYeb/BalatroEval) | 规则可控的游戏仿真与 Agent 决策评价 | 主线持续开发；优先保证端到端运行、Trace 可回放和规则正确性。 |
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | MCP Server / Client 标准实现 | 对照协议与工具调用边界，不要只验证 server 能启动。 |
| [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) | 任务、Solver、Tool、Scorer 组成的评价框架 | 阅读其 task/solver/scorer 组织方式，选取思想而非整体迁移。 |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 标准语言模型 benchmark 框架 | 学习任务定义、模型适配、指标聚合和可复现评价；它主要面向标准化模型任务，不等同于长时序 Agent 评价。 |
| [MCP Servers](https://github.com/modelcontextprotocol/servers) | MCP 工具服务参考 | 用于理解工具暴露方式、输入输出和错误处理。 |

* **辅助教程映射（Runoob）**：

| 教程 | 学习范围 | 使用方式 |
| :--- | :--- | :--- |
| [FastAPI](https://www.runoob.com/fastapi/fastapi-tutorial.html)、[Python 3](https://www.runoob.com/python3/python3-tutorial.html)、[Linux](https://www.runoob.com/linux/linux-tutorial.html) | API 服务、异步调用相关 Python 基础、进程与日志排查 | 结合真实模型调用、工具服务和 Episode Runner 学习 |
| [SQL](https://www.runoob.com/sql/sql-tutorial.html)、[SQLite](https://www.runoob.com/sqlite/sqlite-tutorial.html)、[Redis](https://www.runoob.com/redis/redis-tutorial.html) | 基础查询、持久化、缓存 | 仅当 Trace/任务/缓存确实需要数据库时再学；先用 JSONL/SQLite 即可 |

* **实验设计与阶段验收清单**：

  * 后续实验方向（端到端稳定后再做，未完成不写成已验证能力）：单步 vs 多步决策、不同 Observation 信息量、Direct prompting vs 显式推理、最终得分 vs 局部决策/Regret、相同初始状态配对比较。

  * 阶段验收：真实模型通过 Host 调 MCP 工具；至少一组固定环境与初始状态可复现；Trace 可定位错误步数；有明确评价指标与确定性验证器；分析 Bad Case 并说明指标能否识别；有重复运行、配对比较或不确定性分析。

* **论文与产出清单**：

  * 必读：[ReAct (2022)](https://arxiv.org/abs/2210.03629)、[AgentBench (2023)](https://arxiv.org/abs/2308.03688)、[WebArena (2023)](https://arxiv.org/abs/2307.13854)、[SWE-bench (2023)](https://arxiv.org/abs/2310.06770)、[τ-bench (2024)](https://arxiv.org/abs/2406.12045)。

  * 选读：[Toolformer (2023)](https://arxiv.org/abs/2302.04761)、[LATS (2023)](https://arxiv.org/abs/2310.04406)。

  * 产出：撰写 Agent Benchmark 设计对比（环境、Observation、Action、终止条件、验证器、指标、可重复性）及 BalatroEval 适用性分析。

### 阶段 5：推理系统与 vLLM

* **目标**：先部署与压测测量，再读源码与针对性优化，不重复造轮子。

* **核心仓库**：

| 仓库 | 主要价值 | 建议使用方式 |
| :--- | :--- | :--- |
| [vLLM](https://github.com/vllm-project/vllm) | 高吞吐 LLM serving，包含 PagedAttention、Continuous Batching、Prefix Caching 等 | **核心参考项目**。先部署小模型、调用 OpenAI-compatible API、跑 benchmark，再跟踪调度和 KV Cache 代码。 |
| [nanochat](https://github.com/karpathy/nanochat) | 含推理 benchmark 脚本和简化推理系统 | 用于理解模型训练与推理评价如何连接。 |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Attention Kernel 优化 | 阅读接口、测试和基准；先理解与 PyTorch SDPA 的差异。 |
| [llm.c](https://github.com/karpathy/llm.c) | C/CUDA 实现 LLM 训练 | 利用已有 C++/CUDA 基础做进阶阅读，不作为 Python/PyTorch 入门替代品。 |

* **辅助教程与书籍映射**：

| 资源 | 范围 / 主题 | 使用方式 |
| :--- | :--- | :--- |
| Runoob [Linux](https://www.runoob.com/linux/linux-tutorial.html)、[Docker](https://www.runoob.com/docker/docker-tutorial.html)、[FastAPI](https://www.runoob.com/fastapi/fastapi-tutorial.html) | Linux 进程/资源、容器基础、API 服务 | 用于部署和排障；CUDA、vLLM 本身必须看官方文档 |
| 《CUDA for LLMs》Ch.1、Ch.5、Ch.6 | 优化栈与自定义 Kernel 时机；Decode 访存特征与 GEMV；GEMV/Softmax/LayerNorm/Top-k/GEMM 优化 | 部署和测量 vLLM 后阅读，理解推理瓶颈与 Kernel 优化（另关联 Ch.8、Ch.9、Ch.12 生态） |

* **实验、约束与阶段验收清单**：

  * 实验清单：建小模型推理服务 → 固定长度测 TTFT、TPOT、E2E → 调并发测吞吐与尾延迟 → 观察显存/KV Cache/OOM → 对比长短输入输出与并发 → 读调度、KV Cache、Prefix Cache 代码 → 发现具体瓶颈后再看量化、CUDA Graph、Speculative Decoding 或 Kernel 优化。

  * RTX 4060 注意事项：核对支持矩阵；若兼容性受限，先用 Transformers / llama.cpp 建基准，再把 vLLM 源码作系统学习对象。

  * 阶段验收：能部署小模型服务并设计可复现实验；解释 TTFT/TPOT/吞吐/显存关系；沿源码说明请求进入调度器、分配 KV Cache 与生成 token 过程。

* **论文与产出清单**：

  * 必读：[FlashAttention (2022)](https://arxiv.org/abs/2205.14135)、[FlashAttention-2 (2023)](https://arxiv.org/abs/2307.08691)、[PagedAttention (2023)](https://arxiv.org/abs/2309.06180)、[Orca (2022)](https://www.usenix.org/conference/osdi22/presentation/yu)。

  * 选读：[Speculative Decoding (2023)](https://arxiv.org/abs/2211.17192)、[AWQ (2023)](https://arxiv.org/abs/2306.00978)、[SmoothQuant (2022)](https://arxiv.org/abs/2211.10438)。

  * 产出：结合 vLLM 推理实验解释调度、KV Cache、Prefill/Decode、TTFT/TPOT，并将论文机制映射到源码模块与实测指标。

### 阶段 6：分布式训练、集合通信与 CUDA 进阶

* **目标**：依托 C++/CUDA 基础开展小规模并行通信与算子分析实验，单卡笔记本不强求多机复现。

* **核心仓库**：

| 仓库 | 主要价值 | 建议使用方式 |
| :--- | :--- | :--- |
| [PyTorch Examples](https://github.com/pytorch/examples) | 包含模型和训练示例 | 阅读分布式训练示例，理解 DDP 的进程启动和梯度同步。 |
| [PyTorch Tutorials](https://github.com/pytorch/tutorials) | 官方分布式与性能教程 | 重点阅读 DDP、FSDP、性能分析相关教程。 |
| [NCCL Tests](https://github.com/NVIDIA/nccl-tests) | NCCL 集合通信基准 | 有多 GPU 环境时运行 AllReduce / AllGather 测试；没有环境时阅读测试参数与带宽计算。 |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | CUDA/Triton 高性能 Attention | 重点阅读 kernel、测试和 benchmark 的对应关系。 |
| [llm.c](https://github.com/karpathy/llm.c) | 原生 C/CUDA 训练实现 | 选择 LayerNorm、GEMM 或 Attention 相关模块，结合已有 CUDA 经验阅读。 |

* **辅助教程与书籍映射**：

| 资源 | 范围 / 主题 | 使用方式 |
| :--- | :--- | :--- |
| Runoob [C++](https://www.runoob.com/cplusplus/cpp-tutorial.html)、[Linux](https://www.runoob.com/linux/linux-tutorial.html)、[CMake](https://www.runoob.com/cmake/cmake-tutorial.html)、[Git](https://www.runoob.com/git/git-tutorial.html) | C++ 语法复习、编译构建、Linux 工具链、版本管理 | 查不熟的构建和工具链知识；CUDA 主线用官方资料与专书（含 CUDA C++ Programming Guide、Nsight 与《CUDA for LLMs》） |
| 《CUDA for LLMs》Ch.2、Ch.3、Ch.6 | 线程/Block/Grid 与 Vector Add；朴素 Transpose/GEMM/Softmax/卷积；Coalescing、Warp Reduction、Tiling、Shared Memory | 阶段 6 核心实践章节；进阶读 Ch.7（WMMA/WGMMA/TCGen）、Ch.10（NCCL/TP/PP）、Ch.11（CUTLASS/CuTe）、Ch.12 |

* **实验、作品集与阶段验收清单**：

  * 最小作品集：Vector Add、Matrix Transpose、Softmax、LayerNorm、GEMM（加 Tiling）、简化 Attention、PyTorch C++ Extension。每个 Kernel 需含多 shape/dtype/边界正确性测试、与 PyTorch/cuBLAS 数值对照、预热后稳定计时、明确输入规模与硬件环境，以及适用的 Nsight Systems / Nsight Compute 分析。

  * 实验清单：单机单卡用 PyTorch Profiler 分析迭代；多卡可用时跑最小 DDP 观察 rank/world_size 与梯度同步；理解/运行 `nccl-tests` AllReduce 记录 `algbw` 与 `busbw`；选小算子建正确性测试、基准与 Nsight 分析；暂不盲目搭多机集群或追 Megatron-LM 规模训练。

  * 阶段验收：解释数据并行梯度同步过程；区分计算与通信耗时；通过 profiler 或 benchmark 找到可量化性能瓶颈。

* **论文与产出清单**：

  * 必读：[ZeRO (2019)](https://arxiv.org/abs/1910.02054)、[Megatron-LM (2021)](https://arxiv.org/abs/2104.04473)、[FlashAttention (2022)](https://arxiv.org/abs/2205.14135)。

  * 选读：[Activation Recomputation (2022)](https://arxiv.org/abs/2205.05198)、[FP8-LM (2023)](https://arxiv.org/abs/2310.18313)。

  * 产出：完成 DP/DDP、FSDP/ZeRO、TP、PP、CP 对比（切分对象、通信模式、显存收益、适用场景）及结合 NCCL/profiler 的通信开销解释。

## 四、近期行动与资源优先级

* **近期两周行动清单（2026-11-26 \~ 2026-12-05，聚焦技术交流与已有工程跑通，不临时突击全量 PyTorch/Transformer/RLHF）**：

  * BalatroEval：\[ \] 查启动方式与 stdio 日志是否污染协议；\[ \] 完成工具发现；\[ \] 调通只读工具；\[ \] 调通有状态动作工具；\[ \] 接入真实模型 API；\[ \] 跑通完整 Episode；\[ \] 生成可回放 Trace；\[ \] 分析 1 个 Bad Case；\[ \] 准备 1 分钟项目介绍与 3 分钟技术介绍。

  * Python 补课：\[ \] 建独立项目与虚拟环境；\[ \] 复习模块、异常、类型注解与日志；\[ \] 理解 `async`/`await` 与异步 API 调用；\[ \] 给现有模块补 `pytest` 测试；\[ \] 能解释 BalatroEval 工具调用数据流。

* **动手仓库优先级（1–10）**：

| 顺序 | 仓库 | 当前用途 |
| :--- | :--- | :--- |
| **1** | [BalatroEval](https://github.com/ChzhanYeb/BalatroEval) | 当前短期主线：完成真实 Agent 最小闭环 |
| **2** | [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | 验证 MCP stdio 和工具调用 |
| **3** | [PyTorch Tutorials](https://github.com/pytorch/tutorials) | 补 PyTorch 实操 |
| **4** | [PyTorch Examples](https://github.com/pytorch/examples) | 独立完成 MLP 训练项目 |
| **5** | [build-nanogpt](https://github.com/karpathy/build-nanogpt) | 从零实现 Transformer |
| **6** | [TRL](https://github.com/huggingface/trl) | SFT / DPO / GRPO 实践 |
| **7** | [torchtune](https://github.com/meta-pytorch/torchtune) | 阅读更系统的后训练 recipes |
| **8** | [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) | 评价框架设计参考 |
| **9** | [vLLM](https://github.com/vllm-project/vllm) | 推理系统主线 |
| **10** | [nanochat](https://github.com/karpathy/nanochat) | 后续串联预训练、微调、评价和推理 |

* **分阶段资源使用优先级**：

| 时间窗口 | 主线与资源优先级 |
| :--- | :--- |
| **现在（2026-11-26 至 2026-12-05）** | 主线项目：BalatroEval；文档/协议：MCP Python SDK 官方文档；Runoob：Python 3、Git、Linux（按现有代码遇到的问题查阅）；论文：ReAct、Toolformer（选读 AgentBench）；暂不启动：CUDA for LLMs 全书、复杂 vLLM 部署、GRPO 实验。 |
| **接下来（2026-12 至 2027-01）** | 主线项目：PyTorch MLP；Runoob：Python 3、NumPy、Pandas、Matplotlib；开源仓库：PyTorch Tutorials、PyTorch Examples；论文：Adam、AdamW（配合深度学习教材学习训练机制）。 |
| **再之后（2027-02 至 2027-04）** | 主线项目：build-nanogpt → 小规模 GPT 训练 → SFT/LoRA → DPO；开源仓库：nanochat、TRL、torchtune；论文：Attention Is All You Need、GPT-3、Chinchilla、LLaMA、InstructGPT、LoRA、QLoRA、DPO。 |
| **后续（2027-04 至 2027-05 及以后）** | 主线项目：vLLM 性能分析、CUDA Kernel 实践；书籍：CUDA for LLMs（按章节映射阅读）；开源仓库：vLLM、FlashAttention、llm.c、NCCL Tests；论文：FlashAttention、PagedAttention、Orca、ZeRO、Megatron-LM。 |

## 五、通用学习方法

* **阅读陌生 GitHub 仓库流程**：

  * 固定步骤：阅读 README → 看目录结构 → 找最小运行路径 → 跟踪数据流 → 阅读测试 → 修改一个小功能 → 写实验记录。

  * 必答五问：解决什么问题？核心抽象是什么？主执行路径在哪里？如何验证正确？能独立复现或改造哪一个模块？

* **阅读论文方法**：

  * 三轮阅读：第一轮（10–20 分钟）看摘要、引言、图表、结论定问题；第二轮（30–60 分钟）看方法、公式、实验设置抓机制；第三轮（按需）结合源码或实验复现关键部分。优先读与当前代码任务直接相关的论文。

  * 记录模板条目：论文、研究问题、现有方法的缺陷、核心思想、关键公式/算法、实验设置、主要结果、适用条件、局限性、对项目的启发、是否做了代码验证。