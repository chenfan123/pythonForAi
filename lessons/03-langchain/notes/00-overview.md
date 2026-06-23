# LangChain 学习总览（基于当前项目实现）

这套笔记不是按吴恩达课程视频里的原始版本来写，而是**以当前仓库里的代码为准**，再回扣字幕中的教学主线。这样学完以后，你能直接在这个项目里继续改、继续跑，而不是学完发现 API 和模型都对不上。

## 1. 先知道这套代码和视频哪里不一样

课程原版大致是：

- 模型：`ChatOpenAI`
- 主要模型：`gpt-3.5-turbo`
- Embedding：OpenAI Embeddings
- 课程年代较早，很多高层 API 还是旧写法

当前项目大致是：

- 模型：`ChatQwQ`
- 主要模型：`qwen3.7-plus`
- Embedding：阿里百炼 `text_embedding_v3`
- 依赖版本：`langchain>=1.0.0`，但很多课程示例仍通过 `langchain_classic` 保留旧链式接口

这意味着你会同时看到两套风格：

1. `langchain_core` / `invoke` / `prompt | model | parser`
2. `langchain_classic` / `LLMChain` / `ConversationChain` / `initialize_agent`

这不是代码写乱了，而是因为项目既想保留课程示例，又已经迁移到较新的 LangChain 依赖。

## 2. 课程章节和项目脚本怎么对应

| 视频章节 | 主题 | 当前项目脚本 |
| --- | --- | --- |
| 第一节 | 课程介绍 | 无直接代码 |
| 第二节 | Models / Prompts / Parsers | `01-model&prompts&parsers.py`、`02-llm生成json.py` |
| 第三节 | Memory | `03-memory.py`、`03-memoryWindow.py` |
| 第四节 | Chains | `04-chains.py`、`05-routerChain.py` |
| 第五节 | Question Answering | `06-questionAndAnswer.py` |
| 第六节 | Evaluation | `07-evaluation.py` |
| 第七节 | Agents | `08-agent.py` |
| 项目额外补充 | LangChain 1.x 新风格 | `01-官网教程.py` |

## 3. 推荐学习顺序

建议按下面顺序学，而不是按文件名机械地跑：

1. 先看 `01-model&prompts&parsers.py`
2. 接着看 `02-llm生成json.py`
3. 再看 `03-memory.py` 和 `03-memoryWindow.py`
4. 然后看 `04-chains.py`
5. 再看 `05-routerChain.py`
6. 接着进入 `06-questionAndAnswer.py`
7. 再看 `07-evaluation.py`
8. 最后看 `08-agent.py`
9. 学完以后，把 `01-官网教程.py` 当成“现代 API 对照版”

## 4. 每节最应该抓住什么

| 主题 | 真正要学的东西 |
| --- | --- |
| 模型与提示词 | LangChain 并不是替代模型，而是把“提示词构造”和“模型调用”标准化 |
| 输出解析 | 让 LLM 输出变成结构化对象，而不只是字符串 |
| Memory | 模型本身无状态，所谓“记忆”其实是把历史重新喂回 prompt |
| Chains | 把多个步骤串起来，让输出自动流向下一个步骤 |
| RAG 问答 | 文档先向量化，再检索相关片段，再交给模型生成答案 |
| Evaluation | 用样例集和 LLM-as-a-judge 评估链路效果 |
| Agents | 让模型自己决定“要不要调用工具、调用什么工具” |

## 5. 当前项目里的关键版本差异

### 差异 1：聊天模型换成了 `ChatQwQ`

大部分示例不再直接使用 OpenAI，而是：

```python
model = ChatQwQ(model="qwen3.7-plus", temperature=0)
```

因此你学习时要把重点放在：

- `invoke` 的输入输出形式
- `ChatPromptTemplate` 怎么和模型配合
- LangChain 抽象层本身做了什么

而不是把注意力放在某个具体模型厂商上。

### 差异 2：很多“旧课程 API”被放进了 `langchain_classic`

例如：

- `ConversationChain`
- `LLMChain`
- `SequentialChain`
- `RetrievalQA`
- `initialize_agent`

这说明当前项目是在**新依赖版本里继续跑旧课程示例**。学习时别把它误认为“LangChain 现在最推荐的唯一写法”。

### 差异 3：Embedding 不再走 OpenAI，而是自己封装 `DashScopeEmbeddings`

`06-questionAndAnswer.py` 和 `07-evaluation.py` 都自己实现了：

- `embed_documents`
- `embed_query`

这是很值得学的一点，因为它说明 LangChain 的 embedding 接口是可以被自定义适配的。

### 差异 4：项目里还额外放了一个现代 API 示例

`01-官网教程.py` 使用的是：

- `create_agent`
- `@tool`
- `response_format=Answer`
- `invoke / stream / batch`

这更接近 LangChain 1.x 官方教程风格，可以拿来和课程代码做迁移对照。

## 6. 运行前准备

### 安装依赖

```bash
make install
```

### 推荐运行方式

```bash
./run.sh lessons/03-langchain/01-model\&prompts\&parsers.py
```

项目里的 `run.sh` 会处理虚拟环境和 Apple Silicon 架构问题，尤其是问答示例里涉及本地依赖时，尽量不要直接裸跑 `python xxx.py`。

### 常见环境变量

```env
DASHSCOPE_API_KEY=你的百炼 Key
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```

如果你要自己扩展 OpenAI 或其他模型，再额外补相应的环境变量。

### 某些脚本可能还需要补充依赖

从当前代码看，下面这些能力值得你在本地环境里额外确认：

- `06-questionAndAnswer.py` 使用 `DocArrayInMemorySearch`，如果运行时报缺包，通常要检查 `docarray`
- `08-agent.py` 依赖 `wikipedia`、`langchain-experimental`
- `01-官网教程.py` 使用 `ollama:qwen2.5:7b`，需要本地 Ollama 服务和对应模型

## 7. 这套笔记怎么用最有效

推荐你每一节都按同一套问题去看代码：

1. 这段代码的输入是什么？
2. LangChain 在中间帮我封装了哪一层？
3. 输出是字符串、消息，还是结构化对象？
4. 如果不用 LangChain，手写的话会多出哪些胶水代码？
5. 这一节在真实项目中最容易出什么坑？

只要这 5 个问题都能答出来，这节基本就不是“看过”，而是“学会了”。

## 8. 后续文档索引

- `01-models-prompts-parsers.md`
- `02-memory.md`
- `03-chains.md`
- `04-rag-qa.md`
- `05-evaluation.md`
- `06-agents.md`
- `07-modern-langchain-api.md`
