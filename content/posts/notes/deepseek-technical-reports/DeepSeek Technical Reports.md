---
draft: false
title: "DeepSeek Technical Reports"
date: 2026-09-19T13:36:00+08:00
lastmod: 2026-09-19T13:36:00+08:00
tags: ["DeepSeek", "LLM", "Machine Learning"]
---

# DeepSeekMath

[arxiv](https://arxiv.org/abs/2402.03300) | [hjfy](https://hjfy.top/arxiv/2402.03300)

这篇提出了 GRPO，围绕 LLM Post Training 中的几种方法对 Math 领域问题的训练效果做了讨论。

# DeepSeek-R1

[arxiv](https://arxiv.org/abs/2501.12948) | [hjfy](https://hjfy.top/arxiv/2501.12948)

![DeepSeek-R1 多阶段训练流程](../assets/Pasted%20image%2020260727210604.png)


# DeepSeek-V3.2

[arxiv](https://arxiv.org/abs/2512.02556) | [hjfy](https://hjfy.top/arxiv/2512.02556)

## DSA

DSA (DeepSeek Sparse Attention) 在标准 Attention 前添加了一个 lightning indexer，用于检索出值得进行 Attention 的 token，只让这部分 token 与 query 做 Attention，这就是稀疏注意力，相当于 DSA 先做了一轮粗筛。

Lightning Indexer 的公式如下：

$$
I_{t,s} =
\sum_{j=1}^{H^I}
w^I_{t,j}
\cdot
\operatorname{ReLU}
\left(
q^I_{t,j}\cdot k^I_s
\right)
$$

接下来逐个拆解各个部分：
lightning indexer 计算 query token $h_t$ 和一个先前的 token $h_s$ 之间的检索分数，$s < q$，$I_{t,s}$ 表示 token t 与 token s 之间的检索分数。
lightning indexer 由多个 head 组成，$H^I$ 表示 indexer heads 的数量，$w_{t,j}^I$ 是各个 head 的权重。设计的初衷是期望每个 head 能用于评估 token 的不同层面的价值，比如检查词义相关性、实体相关性、局部依赖、长距离依赖……当然实际训练出来的 head 不会有如此理想的人类语义，这种分 head 的思路其实和 MHA 以及 MoE 这些方法的思想是同构的。
使用 $\text{ReLU}$ 作为激活函数主要是考虑到效率。
indexer 有类似于 Attention 机制的 key/value，注意到公式中的 $q$ 和 $k$ 右上角都用 $I$ 标识它是 indexer 的参数，防止与 Attention 混淆。$q_{t,j}^I, k_{s}^I \in \mathbb{R}^{d^I}$ ，每个 query token 生成 $H^I$ 个 $q$ 参数，每个先前的 token 对应一个 $k$ 。

得到当前 query token 与先前所有 token 之间的 index score 后，从中选出 TopK 的 token 进入 Attention 的计算，这就是 DSA。

$$
u_t =
\operatorname{Attn}
\left(
h_t,
\{c_s \mid I_{t,s}\in \operatorname{Top-k}(I_{t,:})\}
\right)
$$

$\{c_s\}$ 是检索出的 TopK token 对应的 KV pairs。

只对 TopK tokens 做 Attention 时，positional embedding 通常是不连续的。由于 token 的相对位置不变，RoPE 下的位置编码以及由此得到的 QKV 也不变；只要 TopK tokens 覆盖了足够大的 Attention Score，这种稀疏化对结果的影响就较小。类似的位置编码问题也出现在 [Prompt Cache: Modular Attention Reuse for Low-Latency Inference](https://arxiv.org/abs/2311.04934) 中。

# DeepSeek-V4

[arxiv](https://arxiv.org/abs/2606.19348) | [hjfy](https://hjfy.top/arxiv/2606.19348)

[huggingface](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main) 上有相对清晰简易的推理实现。

## Architecture

### mHC

mHC（Manifold-Constrained Hyper-Connections）是在 Hyper-Connections 的多残差流机制上增加几何约束：它让多条 residual streams 之间可以通过可学习矩阵动态混合，从而获得比普通残差连接更灵活的信息路由能力；但同时把负责 residual mixing 的矩阵限制在 doubly stochastic matrix 构成的 Birkhoff polytope 上，使每行、每列元素和都为 1，从而避免深层堆叠时信号被持续放大或衰减。简言之，mHC 的目标是在保留 HC 多路径表达能力的同时，恢复普通 residual connection 那种稳定的信息与梯度传播特性。

### Hybrid Attention

#### CSA

Compressed Sparse Attention
![DeepSeek-V4 CSA 核心架构](../assets/Pasted%20image%2020260915161544.png)
Lightning Indexer 和 Top-k Selector 的部分就是 DSA。Sliding Window KV Entries 是未压缩的 recent tokens，DeepSeek-V4 中 $n_w=128$。Indexer 使用的 KV 与 Attention KV 采用相同的压缩方式，序列位置仍然一一对应，因此可以依据 index score 进行选择。

Compressed KV Entries 是经过有损压缩的 hidden states，不能直接逆向对应到某个特定 token，但可以通过专门训练的 probe 在一定程度上还原语义。稀疏化需要在训练和推理阶段保持一致，让模型在训练中适应这种结构。

#### HCA

Heavily Compressed Attention

HCA 相比 CSA 删去了 Sparse 部分，但提高了压缩率，即高压缩+全量注意力。HCA 的核心目标是建立低分辨率的全局视野。对于所有历史区域，都至少存在某个 compressed entry，而 query 对所有这些 entry 都做 attention。二者不在运行时动态权衡，而是预先定义在模型架构上，在 Transformer Layers 中交错使用。
HCA 用极强的信息压缩消灭 Top-k routing 的 coverage blind spot，让每隔若干层，所有历史区域都重新拥有一条通向当前 token 的 attention 路径。

总的来说，DSV4 的 Attention 机制用 Sliding Window 保留了局部细节，CSA 保证了远程检索精确度，HCA 保证了全局覆盖率/召回率。

### Muon Optimizer

[Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)
[Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)

Muon (MomentUm Orthogonalized by Newton-Schulz) 主要用于神经网络中的二维权重矩阵，例如 Transformer 的线性层，而 embedding、bias、norm 参数以及通常的输出 head 往往仍然交给 AdamW。

## Post Training

OPD 的作用是将多个 expert 的知识蒸馏到最后统一的模型参数中。


# DeepSeek-V4.1

[huggingface](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)

## Architecture


![DeepSeek-V4.1-Flash 总体架构](../assets/Pasted%20image%2020260910153434.png)

### Engram

[Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/abs/2601.07372)

- [知乎解读](https://zhuanlan.zhihu.com/p/1998076357154989024)
- [Bilibili 讲解](https://www.bilibili.com/video/BV1x3zWB6EU6)

### DSpark

[DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147)
在 Abstract 中学到一个新的术语，[Pareto frontier](https://en.wikipedia.org/wiki/Pareto_front)

在多目标优化（multi-objective optimization）中，Pareto 前沿（Pareto front，也称 Pareto frontier 或 Pareto curve）是所有 Pareto 有效（Pareto efficient）解所组成的集合。

直观地说，当一个优化问题需要同时考虑多个不同目标时，Pareto 前沿表示这样一组解：由于不同目标之间存在权衡关系，这组解中的任何一个解都不能在所有方面被另一个解超越；与此同时，那些已经被其他解全面优于的解则会被排除在这个集合之外。

这种“超越”关系称为 Pareto 支配（Pareto dominance）：如果解 A 在每一个目标上都不比解 B 差，并且至少在一个目标上优于解 B，那么就称 A 支配（或优于）B。

这一概念在工程领域中被广泛使用。它使设计者可以把注意力集中在有效解集合上，并在这些有效方案之间进行权衡，而不必考察所有参数可能形成的全部方案。

---

#### Micro-average 与 macro-average

一般意义上的 average 就是指 micro-average，而 macro-average 是宏（观）平均，先对每个子任务、数据集或类别分别计算指标，再对各个结果做**等权**平均。相对的，如果是以所有样本总数为分母对各个类别做加权平均就是 micro-average。

---

ParallelSpec: Parallel Drafter for Efficient Speculative Decoding 是最早提出 parallel drafter 的工作。DFlash 把 block diffusion 引入 speculative decoding，用 diffusion-style drafter 做 single-pass parallel drafting。

![DSpark 架构与解码流程](../assets/Pasted%20image%2020260918151051.png)

#### Semi-autoregressive generation

The average latency per generated token is:

$$L=\frac{T_{draft}+T_{verify}}{\tau}$$

$T_{draft}$ 是 draft model generation 的时间
$T_{verify}$ 是 target model verification 的时间
$\tau$ 是被接受的 token 数
$\gamma$ 是 draft model 生成的 token 数

为了降低延时，有三种方法，降低 $T_{draft}$ ，降低 $T_{verify}$ ，增大 $\tau$ 。面对这个优化目标，触及了 Pareto Frontier，也可以看作是一种不可能三角。

现在 parallel 的方法能在不增加 $T_{draft}$ 的前提下增大 $\gamma$ ，但由于是 non-autoregressive model，它对上下文间的建模能力不够强，导致 $\tau$ 的劣化。为了提高 draft 被 target model 接受的概率，又引入一个 sequence block 用于建模 token-wise 的联系，从而在 $T_{draft}$ 和 $\tau$ 之间取得一个良好的 tradeoff，最终降低整体延时，拓展 Pareto frontier。本文称其为 semi-autoregressive generation。
