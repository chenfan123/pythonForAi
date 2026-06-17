# Transformer LLM:

由 3 个主要部件组成：Tokenizer、Stack of Transformer Blocks、LM Head。

- Tokenizer：拥有一定的词汇量
- Stack of Transformer Blocks：为每个 token 配备了相关的 embeddings 信息。
- LM Head：语言模型处理过程的最后阶段，从 embeddings 信息中预测下一个 token。

## Transformer Blocks 运行流程

1. 被 tokenizer 处理后的每个 token 都会被转换为对应的 embeddings 信息，这些 embeddings 信息会被传递给 Transformer Blocks。
2. 首先这些向量会到第一个变压器模块，该模块会处理这些向量，并生成新的向量。
3. 这些新的向量会传递给第二个变压器模块，该模块也会处理这些向量，并生成新的向量。
4. 这个过程会重复多次，直到所有向量都被处理完毕。
5. 最后，这些向量会传递给 LM Head，该模块会输出或者生成下一个要处理的 token。

### transformer Block

由 2 个主要部分构成：Self-Attention Layer(自注意力层)、Feed-Forward Neural Network(前馈神经网络)。
前馈神经网络：可以看作是一种用于存储信息和统计数据的工具，比如说输入标记之后的下一个单词。

#### Self-Attention Layer

让模型在对当前 token 进行处理时，能够关注到之前的 token，并融入上下文。首先就是相关性评分(Relevance scoring)，衡量相关程度。然后整合信息（Combining information）。
使用三个矩阵来构建：查询投影矩阵、关键投影矩阵、值投影矩阵。这些矩阵用于计算查询键的值以及价值矩阵。
我们拥有的每一个标记，以及分配给他们的所有分数，都在告诉我们这个 token 与我们当前所代表的 token 的相关性。
使用这些分数，我们可以计算出每个 token 的权重，然后使用这些权重来计算出每个 token 的加权平均值。

### 分组查询注意力机制（Grouped Query Attention）

### 稀疏注意力机制（Sparse Attention）

## Recent improvements

下面是 transformer 的简化示意图：
![Recent improvements](https://cdn.heritcoin.com/sky/official/d/image/20260617/c164730qo8yqatprah-W700H1014.png)

左边是编码器，右边是解码器，但是现在大多数模型都是属于解码器类型，不包含编码器组件。

### 旋转 embedding 技术

#### 训练问题

第一步是基础训练，主要对下一次的生成进行训练。也就是语言建模。

### transformer 混合式方法（Mixture of Experts）MoE

这是利用多个子模型提高大语言模型质量的概念。通过引入动态选择的 transformers 来扩展传统的数据迁移方法。

2 个主要组成部分：

- experts
- router

Mixture of Experts 改变了 transformer 模型中 decoder 部分的构造。
由多个神经网络组成，每个神经网络被称为一个专家（expert）。
数据经过一个专家层（MoE 层）之后，会选出一名或者多名专家来处理数据。其他专家处于未激活状态，这种叫做**稀疏模型**，因为只有一部分专家考虑在内，在特定时间被激活。

如何判断输入应该给哪个专家？
使用一个路由器（router）来决定输入应该给哪个专家。本身是一个前馈神经网络，仅仅将输入数据进行路由。
路由器会生成一个概率评分，表明事件发生的可能性有多大，这个专家适合提供哪种特定的信息，根据评分来选择专家。一种方式是选择概率最高的那位；其他策略可以引入一些功能，让输出结果更有创意。
