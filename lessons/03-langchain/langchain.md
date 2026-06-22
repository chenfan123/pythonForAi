模型：

- models - 语言模型
- prompts - 创建用于输入到模型中的数据风格
- output parsers - 解析器：涉及获取这些模型的输出结果，解析成更有结构化的格式，可以用来处理后续的操作

LangChain: 提供了一组易于使用的抽象概念来操作这些结果。

### Memory

#### ConversationBufferMemory 对话缓冲区内存

最简单的记忆类型，对话缓冲区内存。完整保存每一轮 input / output

```python
from langchain_classic.memory import ConversationBufferMemory
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=model,
    memory=memory,
    verbose=True,  # 打印完整 prompt，方便观察 memory 如何被注入
)
conversation.predict(input="Hi, my name is Andrew")  # 第 1 轮：自我介绍
# save_context 接收一对 input / output，格式与 ConversationChain 内部写入的一致
memory.save_context({"input": "Hi"}, {"output": "What's up"})
```

#### ConversationWindowMemory 对话窗口内存

对话窗口内存。只保存最近的对话，超出窗口大小的对话会被丢弃。

```python
from langchain_classic.memory import ConversationWindowMemory
memory = ConversationWindowMemory(k=2) # 只保存最近的2轮对话
memory.save_context({"input": "Hi, my name is Andrew"}, {"output": "What's up"})
memory.save_context({"input": "What's up"}, {"output": "I'm good, thank you"})
memory.save_context({"input": "how are you"}, {"output": "I'm fine,thank you"})
print(memory.load_memory_variables({})) # 以字典形式读取，键名默认为 "history"，供 prompt 模板占位符使用,只有最后2轮的.{'history': "Human: What's up\nAI: I'm good, thank you\nHuman: how are you\nAI: I'm fine,thank you"}
```

#### ConversationBufferTokenMemory 对话令牌缓冲区内存

对话令牌缓冲区内存。按 token 数截断，超出 max_token_limit 时丢弃最早的消息

```python
from langchain_classic.memory import ConversationTokenBufferMemory
from langchain_openai import ChatOpenAI
token_counter = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
memory2 = ConversationTokenBufferMemory(llm=token_counter, max_token_limit=100)
memory2.save_context({"input": "Hi, my name is Andrew"}, {"output": "What's up"})
memory2.save_context({"input": "What's up111"}, {"output": "I'm good, thank you222"})
memory2.save_context({"input": "how are you"}, {"output": "I'm fine,thank you"})
print(memory2.load_memory_variables({}))

```

#### ConversationSummaryMemory 对话摘要内存

对话摘要内存。只保存对话的摘要，超出窗口大小的对话会被丢弃。

```python
from langchain_classic.memory import ConversationSummaryMemory
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."
memory = ConversationSummaryMemory(llm='gpt-3.5-turbo', max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, {"output": f"{schedule}"})
print(memory.load_memory_variables({}))
conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=True
)
conversation.predict(input="What would be a good demo to show?")
print(memory.load_memory_variables({}))
```

更好的方案是使用向量数据库，将对话历史存储到向量数据库中，然后根据查询条件从向量数据库中查询相关的对话历史。

### Chains

指的是可以对大量输入数据进行处理的链式结构。

#### LLMChain

是构成其他 Chain 的基础，把「提示词模板 + 大模型」打包成一个可重复调用的单元。

#### SimpleSequentialChain

用来把多个 Chain 按顺序串起来的容器：前一个 Chain 的输出，自动作为下一个 Chain 的输入。

#### SequentialChain

用来把多个 Chain 按顺序串起来的容器：前一个 Chain 的输出，自动作为下一个 Chain 的输入。

#### RouterChain

用来根据输入决定要使用哪个 Chain。

### 问答系统（向量）

embeddings:将文本片段生成数值化的表示形式，以便于计算相似度。
向量数据库（vector database）：存储和管理向量化的文本数据，以便于快速检索。通过向向量数据图添加一些文本块来创建。当遇到很大的输入文件时，首先会拆分成更小的部分。然后给这些片段分别创建一个嵌入结构。然后存储在向量数据库中。当收到查询请求时，会创建一个实例，然后根据查询请求和向量数据库中的嵌入结构计算相似度，然后返回最相似的文本块。

#### additional methods

1. Map_reduce: 将输入拆分成多个部分，然后分别处理，最后将结果合并。
2. Refine: 将输入拆分成多个部分，然后分别处理，最后将结果合并。输入是上一步的输出，输出是最终的答案。
3. Map_rerank: 将输入拆分成多个部分，然后分别处理，最后将结果合并。
