# Memory：让模型“看起来记得你说过什么”

这一节对应：

- `03-memory.py`
- `03-memoryWindow.py`

课程字幕里的核心观点非常重要：

> 大模型本身并不会记住上一次对话，所谓 Memory，本质上是把历史内容重新组织后再塞回 prompt。

只要你理解了这句话，这一节就不会学偏。

## 1. 为什么需要 Memory

如果没有 Memory，多轮对话其实是这样的：

1. 你问一句
2. 模型单次回答
3. 下一轮请求重新开始

模型不会天然知道：

- 你前面介绍过自己是谁
- 你刚才问过什么问题
- 上一轮回答里埋了什么上下文

所以聊天机器人、客服机器人、Copilot 类产品，都需要某种形式的“记忆注入”。

## 2. `03-memory.py`：最基础的完整对话记忆

这个文件主要演示 `ConversationBufferMemory`。

### 核心组合

```python
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=model,
    memory=memory,
    verbose=True,
)
```

它的意思可以直接翻译成：

- `ConversationChain` 负责多轮对话流程
- `ConversationBufferMemory` 负责存储完整历史
- 每次 `predict()` 时，LangChain 都会把历史注入到 prompt 里

### 关键观察点

#### 观察点 1：模型能回答“我叫什么”

```python
conversation.predict(input="Hi, my name is Andrew")
conversation.predict(input="What is 1+1?")
conversation.predict(input="What is my name?")
```

第三轮能回答名字，不是因为模型真的长期记住了，而是因为第一轮内容还在 memory 里。

#### 观察点 2：`verbose=True` 很适合入门

这一节强烈建议保留 `verbose=True`，因为你会直接看到：

- LangChain 生成了什么 prompt
- 历史是怎么拼进去的
- 记忆变量最终长什么样

#### 观察点 3：`memory.buffer` 是最直观的调试入口

```python
print(memory.buffer)
print(memory.load_memory_variables({}))
```

这两种看法的区别是：

- `memory.buffer`：更像人类读的完整文本
- `load_memory_variables({})`：更像 prompt 模板要消费的变量字典

## 3. 手动写入 Memory 的意义

在 `03-memory.py` 后半段，代码用到了：

```python
memory.save_context({"input": "Hi"}, {"output": "What's up"})
```

这一步特别值得注意，因为它说明：

- Memory 不一定非要来自实时对话
- 你也可以从数据库、日志、历史工单中恢复上下文
- 然后再把这些历史喂给模型

这也是很多真实业务场景会做的事情。

## 4. `03-memoryWindow.py`：不同的记忆策略

项目没有停在“完整保存历史”，而是进一步演示了三种常见策略：

1. 按轮数截断
2. 按 token 数截断
3. 用摘要压缩历史

## 5. `ConversationBufferWindowMemory`：只保留最近几轮

```python
window_memory = ConversationBufferWindowMemory(k=2)
```

这意味着只保留最近 2 轮对话。

适合场景：

- 最近上下文最重要
- 老对话价值不高
- 你想控制 prompt 长度和成本

当前项目和视频还有一个小差异：

- 视频里常说 `ConversationWindowMemory`
- 当前代码实际使用 `ConversationBufferWindowMemory`

这属于 API 演进后的命名差异，理解思想比死记类名更重要。

## 6. `ConversationTokenBufferMemory`：按 token 限制历史

```python
token_counter = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
token_memory = ConversationTokenBufferMemory(llm=token_counter, max_token_limit=100)
```

这里是项目里一个非常有价值的“版本现实问题”：

- 课程视频默认用 OpenAI，token 计数链路比较顺
- 当前项目主模型是 `ChatQwQ`
- 但 `ChatQwQ` 没实现 `get_num_tokens_from_messages()`

所以代码采取了一个折中方案：

- 主回答模型继续用 `ChatQwQ`
- token 计数单独借助 `ChatOpenAI`

注意这里的 `ChatOpenAI` 主要用于**本地 token 计数能力**，不是拿来替换业务主模型。

这说明一件很现实的事：

> 换模型供应商以后，LangChain 的高层抽象并不一定 100% 等价可替换，某些工具能力仍然会有缺口。

## 7. `ConversationSummaryMemory`：用摘要压缩历史

```python
summary_memory = ConversationSummaryMemory(llm=model)
```

这种做法不是存完整对话，而是不断生成“到目前为止的摘要”。

优点：

- 更省上下文窗口
- 适合超长对话

缺点：

- 会丢细节
- 摘要如果写偏，后续对话就可能沿着错误摘要继续走

项目里还有一个很好的提醒：

```python
summary_memory = ConversationSummaryMemory(llm=model)  # llm 必须是模型实例，不能传字符串
```

这就是实战里常见的坑，很多人会误写成 `"gpt-3.5-turbo"` 这种字符串。

## 8. 这节背后的统一思路

无论是哪种 Memory，底层逻辑都一样：

1. 保存历史
2. 在下一轮组装 prompt
3. 把历史作为上下文重新发给模型

变化的只是“保存什么”和“保存多少”。

## 9. 当前项目相对视频的最大学习价值

这一节最值得你借鉴的，不是类名，而是它把“课程概念”落到了“项目现实”：

- 用 `langchain_classic.memory` 承接旧课程 API
- 用 `ChatQwQ` 作为当前主模型
- 在 token memory 上显式处理模型能力差异

这比只抄视频代码更接近真实开发。

## 10. 真实项目里怎么选记忆策略

### 选完整缓冲区

适合：

- 对话较短
- 需要完整保留上下文
- 调试阶段

### 选窗口记忆

适合：

- 用户问题高度依赖最近几轮
- 你想控制成本

### 选摘要记忆

适合：

- 对话非常长
- 更关心整体主题，不关心逐句细节

### 选 token 记忆

适合：

- 你需要精确控制上下文大小
- 不同模型上下文窗口差异明显

## 11. 建议你自己动手改的练习

1. 把 `ConversationBufferMemory` 改成窗口记忆，比较“问名字”是否还能成功。
2. 把 `k=2` 改成 `k=1`、`k=3`，观察 `history` 的变化。
3. 在摘要记忆里加入更长的对话，观察摘要会不会丢掉关键事实。

## 12. 学完这节你应该能回答

1. 为什么说 Memory 不是模型真正记忆，而是 prompt 注入？
2. `memory.buffer` 和 `load_memory_variables({})` 有什么区别？
3. 为什么当前项目的 token memory 需要借助 `ChatOpenAI`？
4. 完整记忆、窗口记忆、摘要记忆分别适合什么场景？
