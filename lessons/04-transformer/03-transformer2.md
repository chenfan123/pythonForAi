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
自注意力机制：让模型在对当前 token 进行处理时，能够关注到之前的 token，并融入上下文。首先就是相关性评分，衡量相关程度。然后是得分之后判断相关性。
