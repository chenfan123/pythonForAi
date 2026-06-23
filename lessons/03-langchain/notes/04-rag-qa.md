# 基于向量检索的问答：RAG 入门

这一节对应 `06-questionAndAnswer.py`，也是整套课程里最接近真实业务场景的一节。

课程字幕里的主线是：

1. 读取外部文档
2. 生成 embedding
3. 建立向量索引
4. 先检索相关片段
5. 再让模型基于这些片段回答问题

当前项目完全保留了这条主线，但把模型和 embedding 都替换成了更符合本仓库环境的实现。

## 0. 运行前提醒

这个示例除了 LangChain 基础依赖外，还隐含依赖向量库实现。如果你运行时提示 `docarray` 缺失，通常需要单独补装。

## 1. 为什么问答系统不能只靠模型直接答

如果只问模型，不接外部文档，会有两个问题：

1. 模型不知道你私有数据里的内容
2. 模型上下文窗口有限，没法一次吃完大量文档

RAG 的基本思路就是：

> 先从文档库里找出“和问题最相关”的少量片段，再把这些片段交给模型作答。

所以它不是“让模型记住所有文档”，而是“让模型在回答前先查资料”。

## 2. 这个脚本的整体流程

当前文件非常适合当作 RAG 的最小教学样例，因为它把流程拆得很完整：

1. 加载 CSV
2. 自定义 embedding 类
3. 创建向量库
4. 做相似度检索
5. 手动拼接检索结果问模型
6. 用 `RetrievalQA` 自动化检索和回答
7. 用 `VectorstoreIndexCreator` 进一步一站式封装

## 3. 第一步：加载文档

```python
loader = CSVLoader(file_path=file)
docs = loader.load()
```

这里的关键不是 CSV，而是 `Document` 这个中间抽象。

LangChain 会把每一行数据转成文档对象，后面无论是向量化、检索还是问答，处理的都是这些 `Document`。

## 4. 第二步：自定义 `DashScopeEmbeddings`

这是当前项目相对视频最重要的改造点之一。

课程原版通常直接用 OpenAI Embeddings，而当前项目自己实现了：

```python
class DashScopeEmbeddings(Embeddings):
```

并补齐了两个核心方法：

- `embed_documents`
- `embed_query`

这说明 LangChain 对 embedding 的要求其实并不复杂：

- 文档入库时，能把文本批量变成向量
- 查询时，能把问题变成单条向量

只要你满足这个接口，就能把自己的 embedding 服务接进 LangChain。

## 5. 为什么这里值得重点学

因为它把“框架抽象”讲透了。

你真正依赖的不是某家云服务，而是：

- `Embeddings` 抽象
- `VectorStore` 抽象
- `Retriever` 抽象

当这三层接口稳定后，模型厂商和 embedding 厂商都可以换。

## 6. 第三步：建立向量库

```python
db = DocArrayInMemorySearch.from_documents(docs, embedding)
```

这里选的是 `DocArrayInMemorySearch`，优点很明显：

- 不需要外部数据库
- 上手快
- 适合教学和原型验证

缺点也很明确：

- 数据不持久化
- 规模大了以后不适合生产

所以你应该把它理解成“演示用向量库”，不是最终生产方案。

## 7. 第四步：先做一次纯检索

```python
docs = db.similarity_search(query)
```

这是 RAG 初学者非常容易跳过的一步，但其实特别重要。

因为 RAG 效果不好时，问题往往不在“生成”，而在“检索”：

- 检索出来的片段不相关
- 召回范围不对
- 文档切分方式不合适

所以在自动问答之前，先单独看检索结果，是很好的调试习惯。

## 8. 第五步：手动 RAG

```python
qdocs = "".join(doc.page_content for doc in docs)
response = model.invoke(
    f"{qdocs}\nQuestion: Please list all your shirts with sun protection ..."
)
```

这段代码的价值在于，它把 RAG 的底层逻辑彻底摊开了：

1. 先检索文档
2. 把检索结果拼进 prompt
3. 再让模型回答

你会发现，RAG 本质上并不神秘。它只是“检索 + Prompt 注入 + 生成”。

## 9. 第六步：`RetrievalQA`

```python
qa_stuff = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
)
```

这一步是把前面的手动流程封装起来。

`chain_type="stuff"` 的意思可以理解为：

- 把检索到的文档都直接塞进 prompt

它最适合：

- 文档片段不多
- 上下文长度还扛得住

如果文档更多，就需要进一步考虑：

- map-reduce
- refine
- rerank

课程字幕在这一节结尾也提到了这些扩展方式。

## 10. 第七步：`VectorstoreIndexCreator`

```python
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embedding,
).from_loaders([loader])
```

这相当于把：

- 加载文档
- 生成向量
- 建立向量库

打包成了一步。

后面你就能直接：

```python
result = index.query(query, llm=model)
```

它很适合快速验证思路，但也容易让初学者看不清底层流程。所以建议你把这一层当成“封装版”，先学明白前面的展开版，再用它。

## 11. 当前项目和视频版本的关键差异

### 差异 1：Embedding 改成百炼接口

这让你能看到真实的供应商适配过程，而不只是调用现成封装。

### 差异 2：加入了 Apple Silicon / Rosetta 保护

脚本开头专门判断了架构问题，提醒你优先通过：

```bash
./run.sh lessons/03-langchain/06-questionAndAnswer.py
```

运行这类依赖较重的脚本。

### 差异 3：项目里同时展示了“手动 RAG”和“框架封装 RAG”

这一点比只看视频更好，因为你既能懂原理，也能懂封装。

## 12. RAG 学习时最容易踩的坑

### 坑 1：把所有问题都归咎于模型

很多时候答案差，是因为没召回到对的文档，不是模型太笨。

### 坑 2：不单独检查检索结果

如果你不看 `similarity_search()` 的输出，就很难判断问题出在检索还是生成。

### 坑 3：把教学型向量库直接当生产方案

`DocArrayInMemorySearch` 很适合入门，但不代表适合线上系统。

## 13. 建议你自己动手改的练习

1. 把查询从英文改成中文，比较检索和回答效果。
2. 把 `OutdoorClothingCatalog_1000.csv` 换成 `OutdoorClothingCatalog_1000_zh.csv`，观察中文数据对问答表现的影响。
3. 在手动 RAG 里打印检索到的每个文档片段，看看哪些片段最有帮助。

## 14. 学完这节你应该能回答

1. RAG 为什么要先检索、后生成？
2. `DashScopeEmbeddings` 这个类到底帮项目补齐了什么能力？
3. 手动 RAG 和 `RetrievalQA` 的本质区别是什么？
4. 为什么 `DocArrayInMemorySearch` 适合教学，但不一定适合生产？
