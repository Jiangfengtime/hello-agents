# Transformer 三种架构对照详解：Encoder-Only、Decoder-Only 与 Encoder–Decoder

[返回第一部分学习指南](01-智能体与语言模型基础详解.md) · [先读 Transformer 架构](08-Transformer架构原理详解.md) · [深入 Decoder-Only](09-Decoder-Only架构详解.md) · [看翻译完整流程](07-Transformer完整流程示例.md)

> 本文以第三章的 Transformer 和 Decoder-Only 内容为基础，回答三个问题：Encoder 和 Decoder 各做什么？三种架构处理输入时有什么本质区别？BERT、GPT、ChatGPT、Bard 分别属于哪个比较层次？文中的分词、概率和模型输出都是教学示意，不是实际运行结果。

## 1. 先分清三个比较对象

**Encoder（编码器）**和 **Decoder（解码器）**首先是 Transformer 的网络模块名称，不是两个独立产品。原始 Transformer 把它们组合起来处理机器翻译：Encoder 处理源语言，Decoder 根据源语言逐步生成目标语言。后来也出现只使用 Encoder 主干或只使用 Decoder 主干的模型。[原始 Transformer 论文](https://arxiv.org/abs/1706.03762)

| 比较层次 | 例子 | 它回答的问题 |
| --- | --- | --- |
| 网络架构 | Encoder-Only、Decoder-Only、Encoder–Decoder | 信息如何流动？哪些位置可见？ |
| 具体模型或模型系列 | BERT、原始 GPT、T5 | 用什么结构、目标和数据训练？ |
| 面向用户的产品 | ChatGPT、Bard／Gemini | 用户通过什么应用与模型交互？ |

因此，讨论 **BERT 和 GPT** 可以重点比较编码器与解码器结构；讨论 **ChatGPT 和 Bard** 时，要先意识到它们是产品名称，不能仅凭产品名断定底层所有模型的内部结构。

## 2. 最核心的区别：当前位置能读取哪些位置

假设输入被分成四个 Token：

```text
位置：  0     1     2     3
Token：我    在    银行   存款
```

在**同一层自注意力**中，位置 2 的「银行」能读到哪些位置？

| 架构中的注意力 | 「银行」所在位置能读到 | 原因 |
| --- | --- | --- |
| Encoder 自注意力 | 我、在、银行、存款 | 整段输入已知，通常不使用因果掩码 |
| Decoder 因果自注意力 | 我、在、银行 | 位置 3 属于未来，因果掩码将其挡住 |

用 `1` 表示允许注意、`0` 表示不允许。忽略 Padding Mask 时，四个位置的可见性可以写成：

```text
Encoder 自注意力             Decoder 因果自注意力
        读 0  1  2  3                读 0  1  2  3
查 0       1  1  1  1         查 0       1  0  0  0
查 1       1  1  1  1         查 1       1  1  0  0
查 2       1  1  1  1         查 2       1  1  1  0
查 3       1  1  1  1         查 3       1  1  1  1
```

**位置信息**和**因果掩码**解决不同问题：位置信息让模型利用顺序与距离；掩码决定某个位置究竟能不能读到另一个位置。它们可以同时存在，不能互相替代。位置向量如何构造，可看[位置编码详解](08-Transformer架构原理详解.md#32-为什么需要位置信息)。

## 3. Encoder-Only：读完整输入，形成上下文表示

### 3.1 数据怎样走

```text
完整输入 → 分词与 Token ID → Token 嵌入 + 位置信息
       → 多层 Encoder：双向自注意力 + 前馈网络
       → 每个输入位置的上下文向量
       → 任务头：分类、抽取、匹配，或预测被遮住的 Token
```

以「我在银行存款」为例，「银行」的初始 Token 嵌入只是一个通用向量；经过 Encoder，它能参考右侧的「存款」以及左侧的「我在」，形成更贴合这句话语境的表示。输出通常不是一句新写出的回答，而是**与输入位置相对应的向量**，随后再由任务头决定怎样使用。

例如情感分类可以从整句表示预测「正面／负面」；命名实体识别可以为输入的每个位置预测标签；检索或匹配任务可以比较文本表示。具体取哪一个位置、怎样汇总以及训练什么任务头，取决于模型和任务，不能把这些方法当成所有 Encoder-Only 模型的固定规则。

### 3.2 BERT 为什么采用「遮住再猜」

BERT 是典型的 Encoder-Only 模型。它的名字是 **Bidirectional Encoder Representations from Transformers**，突出「双向」与「编码器」。[BERT 论文](https://arxiv.org/abs/1810.04805) · [Google 对 BERT 的解释](https://research.google/blog/open-sourcing-bert-state-of-the-art-pre-training-for-natural-language-processing/)

教学示意：

```text
原句：我 在 银行 办理 存款
输入：我 在 [MASK] 办理 存款
目标：根据左右两侧的内容，预测被遮住的位置是「银行」
```

上例展示的是被替换成 `[MASK]` 的情况，使模型不能直接复制该位置的答案。不过原始 BERT 并非把所有预测目标都变成 `[MASK]`：它选取约 15% 的 Token 位置作为 MLM 目标，其中约 80% 替换为 `[MASK]`、10% 替换为随机 Token、10% 保持原样；MLM 损失只在被选中的目标位置计算。保留和随机替换的一部分用于缓解预训练与下游使用时没有 `[MASK]` 的差异，不能把“每个目标都必须被遮住”当成 BERT 的完整训练规则。[BERT 原始论文](https://arxiv.org/html/1810.04805v2)

原始 BERT 还使用下一句预测（NSP）辅助目标，其输入包含 Token、可学习的位置和句段类型嵌入。后来的 Encoder 模型可能改变这些设计，因此 BERT 的具体训练方案不等于所有 Encoder-Only 模型的定义。做分类等下游任务时通常直接输入完整文本，在表示上接任务头，不需要先把用户句子随机挖空。

**容易误解的地方**：Encoder-Only 并非完全不能输出文字。BERT 可以预测被遮住位置的 Token；只是它的基础预训练方式和结构，并不是像自回归聊天模型那样天然按「前缀 → 下一个 Token → 再接回前缀」连续写长回答。

## 4. Decoder-Only：读已有前缀，预测下一个 Token

### 4.1 数据怎样走

```text
提示词 + 已生成内容 → 分词与 Token ID → Token 嵌入 + 位置信息
                  → 多层 Decoder 风格模块：因果自注意力 + 前馈网络
                  → 最后一个可见位置的表示 → 整个词表的分数／概率
                  → 选出下一个 Token → 接回序列 → 重复
```

设当前前缀为「我在银行」，模型会估计：

```text
P(下一个 Token | 「我在银行」)
```

如果选中「办」，下次再根据「我在银行办」预测。真实分词器未必把「办理」拆成「办」「理」；这里只说明生成循环。模型在每一步看到的是**整个已知前缀**，并非只能看到最后一个词。

Decoder-Only 中的「Only」表示**没有单独的 Encoder**。输入提示和输出通常进入同一条 Token 序列；基础文本结构也不需要原始翻译模型那种「读取 Encoder 输出」的交叉注意力。它依然能利用提示中的信息来回答问题。

### 4.2 训练为什么也可以一次处理多个位置

设训练文本是「今天 天气 很 好」。完整文本可以作为一批输入送进网络，同时使用因果掩码限制每个位置：

| 输入位置当前可见的前缀 | 该位置要预测的下一个 Token |
| --- | --- |
| 今天 | 天气 |
| 今天 天气 | 很 |
| 今天 天气 很 | 好 |

模型在**训练时**可以并行计算这些位置的预测；在**生成时**，下一位置的输入依赖刚生成的 Token，所以逐步推进。这两个阶段使用的是同一类因果约束。具体的标签错位、掩码与生成循环可看[Decoder-Only 架构详解](09-Decoder-Only架构详解.md)。

## 5. Encoder–Decoder：先编码源输入，再生成目标输出

Encoder–Decoder 有两条序列。以英译中为例：

```text
源序列：I love cats
             ↓
Encoder：读取完整英文句子，得到每个源位置的表示 H
             │
             └───────────────┐
                             ↓ 交叉注意力读取 H
目标前缀：<BOS> 我 爱 → Decoder → 预测下一中文 Token「猫」
                                      ↓
                                  接回目标前缀，继续生成
```

Decoder 内部有两类注意力，需要分开看：

1. **因果自注意力**：目标位置只能读取已经给出的中文目标前缀，不能读取后面的目标词。
2. **交叉注意力**：当前目标位置读取 Encoder 对完整英文源序列的表示。常见写法中，查询 `Q` 来自 Decoder 当前状态，键 `K` 和值 `V` 来自 Encoder 输出。

这里「目标侧不能看未来」与「可以看完整源句」没有冲突：被挡住的是**未来目标词**，而源句从一开始就已经给出。训练时，目标序列通常右移后输入 Decoder，让当前位置预测下一个目标 Token；推理时则从起始 Token 开始逐步生成。

项目的 [`Transformer.py`](../../code/chapter3/Transformer.py) 实现了这一路径：`EncoderLayer` 处理源序列，`DecoderLayer` 中的 `self_attn` 处理目标前缀，`cross_attn(x, encoder_output, encoder_output, src_mask)` 让目标侧读取源侧表示。该文件演示一次前向传播；**它整体是 Encoder–Decoder 示例，不是完整的 Decoder-Only 聊天模型**。若想逐步追踪翻译中的每一个环节，看[Transformer 完整流程示例](07-Transformer完整流程示例.md)。

## 6. 三种架构放在一起比较

| 问题 | Encoder-Only | Decoder-Only | Encoder–Decoder |
| --- | --- | --- | --- |
| 是否有独立 Encoder？ | 有，作为主体 | 没有 | 有 |
| 是否有独立 Decoder？ | 没有 | 只有因果生成主干 | 有 |
| 主体自注意力能看什么？ | 完整输入中可见的各位置 | 当前及之前的位置 | Encoder 看完整源序列；Decoder 看目标前缀 |
| 需要 Encoder–Decoder 交叉注意力吗？ | 不需要 | 基础结构不需要 | 需要，目标侧读取源侧 |
| 常见输出形式 | 输入位置／整句的表示，经任务头输出结果 | 反复预测下一个 Token | 在源输入条件下反复预测目标 Token |
| 代表性模型 | BERT | 原始 GPT 系列 | 原始 Transformer、[T5](https://arxiv.org/abs/1910.10683) |

这张表描述的是**典型基础结构**，不是能力的绝对边界：Decoder-Only 也能阅读、分类和总结输入；Encoder-Only 加上合适的任务头能预测文字；现代模型还可能改动注意力形式、位置表示、归一化方法或多模态处理方式。

### 同样是「翻译」，为什么两种生成架构都能做？

- **Encoder–Decoder**：源句走 Encoder，目标前缀走 Decoder；Decoder 通过交叉注意力反复查询源句表示。
- **Decoder-Only**：把「请把 I love cats 翻译成中文：」作为提示放在同一序列前部，然后接着生成译文。

因此，能做同一种任务，不代表网络内部的信息流相同。选择架构还涉及训练目标、数据、推理方式和工程权衡。

## 7. BERT、GPT、ChatGPT、Bard 到底怎样对应？

### 7.1 如果你想到的是 BERT

这是**模型架构层面**的经典比较：

```text
BERT：Encoder-Only → 双向读取输入 → 形成上下文表示
GPT 类：典型 Decoder-Only → 因果读取前缀 → 自回归生成
```

Google 对 BERT 的原始介绍，明确把 BERT 的深层双向表示与早期 GPT 的单向表示作了比较。[Google 对 BERT 与 GPT 的比较](https://research.google/blog/open-sourcing-bert-state-of-the-art-pre-training-for-natural-language-processing/)

但是，**「BERT 用来理解、GPT 用来生成」只是帮助入门的概括**，不应推导出「GPT 完全不理解」或「BERT 绝对不能预测任何文本」。真正的结构差异是可见性、训练目标和常见输出方式。

### 7.2 如果你想到的是 Bard

这是**产品层面**的比较：ChatGPT 是 OpenAI 的聊天产品；Bard 曾是 Google 的聊天产品，并于 2024 年更名为 Gemini。Google 的说明也把 Gemini 聊天体验与其底层模型系列区分开来。[Google：Bard 更名 Gemini](https://blog.google/intl/en-africa/products/explore-get-answers/bard-becomes-gemini-try-ultra-10-and-a-new-mobile-app-today/)

所以「ChatGPT 是 Decoder-Only，Bard 是 Encoder-Only」不是一个严谨的对照。Bard／Gemini 不是 BERT 的别名，也不能仅凭聊天产品名判断其某个版本底层模型的精确网络结构。

对 ChatGPT 也应保持同样的层次区分：**GPT 代表一类生成模型的历史与技术脉络，ChatGPT 指产品体验**。官方 OpenAI 文档说明 GPT 文本生成模型可以根据提示产生文本，也把 ChatGPT 作为文本生成应用的例子；这些产品文档并未据此逐一公开 ChatGPT 所有现用模型的完整内部结构。因此，本篇用「GPT 类模型通常采用 Decoder-Only」解释基本原理，**不把它当成所有 ChatGPT 模型的精确架构声明**。[OpenAI Docs：基础概念](https://developers.openai.com/api/docs/concepts) · [OpenAI Docs：文本生成](https://developers.openai.com/api/docs/guides/text)

## 8. 三道自测题

1. 一个模型处理输入中间的某个词时，同时利用左边和右边内容，随后输出每个词的位置表示。这最像哪类架构？
2. 一个模型把用户问题和已生成答案放进同一序列，反复预测下一个 Token；它一定有独立的 Encoder 吗？
3. 翻译模型的 Decoder 不能看未来的中文目标词，却能参考完整英文源句。完整英文来自哪里？通过什么注意力读取？

<details>
<summary>查看答案</summary>

1. Encoder-Only 的典型处理方式。
2. 不一定；这正是基础 Decoder-Only 的常见做法，它没有独立 Encoder。
3. 完整英文先经 Encoder 得到源侧表示，再由 Decoder 的交叉注意力读取。

</details>

## 9. 一句话记忆

> **Encoder** 让已知输入的各位置相互提供上下文；**因果 Decoder** 只凭已知前缀继续预测；**Encoder–Decoder** 则让目标侧在逐步生成时持续读取已编码的源输入。
