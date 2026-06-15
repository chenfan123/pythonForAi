# Transformer

最初的的 2 个主要部分组成：编码器（encoder）和解码器（decoder）。

编码器：对整个输入文本进行预处理，提取相关上下文信息
解码器：利用编码器提供的上下文信息来生成文本

## Transformer 大语言模型如何工作

### 词袋模型（Bag of Words）

是一种用**大型稀疏向量**或**数字数组**来表示单词的算法,只关心一个词有没有出现，出现几次，不理解词义。

标记化（Tokenization）：把人类输入的文本，切分成模型能处理的最小单位 token。

token: 模型能理解的最小单位，通常是单词或子词。

词表（vocabulary）：模型 tokenizer 认识的所有 token 集合，记录了 token 和对应的数字 ID，tokenizer 会用它来查 ID

向量表示法（Vector Representation）：把 token 转换为数字向量，便于模型处理。

### Word2Vec（word embeddings 的一种方法）

词袋法的缺陷：认为语言是一种完全被词汇直接描述的工具，忽略了语义和含义。

Word2Vec：把单词变成带有语义关系的数字向量。

本质运用了神经网络技术，通过大量语料库训练，让模型学习到单词之间的语义关系。

![Word2Vec](https://cdn.heritcoin.com/sky/official/d/image/20260615/c170547do80da7vcat-W2054H634.png)

- **word embedding**：将单词转换为向量表示的方法。模型会为每个单词/token 分配一组固定长度的数值，这些数值通常先随机初始化，再通过训练逐步学习出语义信息。

![嵌入](https://cdn.heritcoin.com/sky/official/d/image/20260615/c17154192j74lrk4ig-W1750H836.png)
**嵌入（Embedding）**：由一组高维度数字向量组成，这些数字向量由模型通过训练学习得出，通常介于-1 ～ 1 之间，一个模型里，一个嵌入所拥有的属性或值的数量被称为维度（dimension）。通常是固定的。比如：`cat  → [0.12, -0.45, 0.08, 0.91, -0.33]`

tokenizers（分词器）：具有固定的词汇表，不能代表所有存在的单词。还负责把 token 映射成模型认识的数字。输出的不仅是 id，还有一些辅助信息，比如：token 的类型、token 的子词、token 的词性等。
