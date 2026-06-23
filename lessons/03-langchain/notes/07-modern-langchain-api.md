# 补充：项目里的现代 LangChain 1.x 写法

这一节不是课程字幕里的原始内容，但它很值得保留，因为 `01-官网教程.py` 展示了一个重要事实：

> 这个仓库一边保留课程里的 `langchain_classic` 写法，一边也已经开始接触 LangChain 1.x 风格。

如果你只看前面的课程脚本，很容易误以为 `LLMChain`、`initialize_agent` 就是今天 LangChain 的全部面貌。这个补充文件正好能帮你建立“旧课程写法”和“较新官方写法”的对照关系。

## 0. 运行前提醒

这个文件使用的是：

```python
model="ollama:qwen2.5:7b"
```

所以它不是“只装 Python 依赖就能跑”的脚本，还要求你本地已经安装并启动了 Ollama，而且对应模型已经准备好。

## 1. 这个文件演示了什么

`01-官网教程.py` 主要演示四件事：

1. 用 `@tool` 定义工具
2. 用 `create_agent(...)` 创建 agent
3. 用 `response_format=Answer` 指定结构化返回
4. 用 `invoke`、`stream`、`batch` 展示不同调用方式

## 2. 先看工具定义

```python
@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"
```

这和 `08-agent.py` 里的 `@tool` 思想一致，但写法更靠近当前 LangChain 官方示例风格。

## 3. 再看结构化输出

```python
class Answer(BaseModel):
    summary: str
    confidence: float
```

然后：

```python
agent = create_agent(
    model="ollama:qwen2.5:7b",
    tools=[search],
    system_prompt="你是一个智能助手",
    response_format=Answer,
)
```

这里很有意思，因为它把两个前面学过的概念结合起来了：

- agent
- 结构化输出

也就是说，现代 API 越来越倾向于把“模型行为约束”直接集成到高层接口里。

## 4. `invoke / stream / batch` 的学习价值

这个文件虽然很多代码注释掉了，但非常适合做对照学习。

### `invoke`

单次请求，最适合入门理解调用结构。

### `stream`

流式输出，适合聊天 UI、长回答、逐 token 展示。

### `batch`

批量处理，适合把多个输入一起送进统一流程。

这提醒你：LangChain 不只是帮你“写 prompt”，它也在逐步统一不同执行模式的接口风格。

## 5. 这个文件为什么和主课程不完全一致

因为它更像“现代补充材料”，而不是“字幕逐字对应代码”。

前面课程文件更多是：

- `langchain_classic`
- 课程迁移版
- 方便理解原理

这个文件更多是：

- LangChain 1.x 官方风格
- 更现代的 agent 创建方式
- 更贴近后续新项目会遇到的接口

## 6. 你应该怎么使用这份补充

推荐的使用方式不是先学它，而是：

1. 先吃透前 6 份课程笔记
2. 再回来看这个文件
3. 思考“如果把经典 API 迁移到现代 API，大概会怎么改”

这样你就不会只停留在“会抄课程代码”，而能开始形成迁移能力。

## 7. 一个很实用的对照表

| 课程脚本里常见 | 现代补充里常见 |
| --- | --- |
| `initialize_agent(...)` | `create_agent(...)` |
| `LLMChain` | `prompt | model | parser` 或更高层封装 |
| `predict` / `run` | `invoke` |
| 经典工具装配 | 更统一的 agent 创建接口 |

## 8. 学完这份补充你应该意识到

1. 当前项目是在“新依赖 + 旧课程 API”之间做兼容
2. LangChain 正在往更统一的接口风格发展
3. 你学习课程代码时，最好同时建立“未来怎么迁移”的视角

这份补充的价值，不是让你立刻重写全部示例，而是让你知道：**你现在学到的是 LangChain 的核心思想，这些思想会跨 API 版本延续下去。**
