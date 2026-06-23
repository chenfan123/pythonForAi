# Agents：让模型决定“下一步该调用什么工具”

这一节对应 `08-agent.py`。

课程字幕里把 agent 描述成“把 LLM 当作 reasoning engine”，这个表述非常到位。因为到了 agent 阶段，模型不再只是回答文本，而是要做三件事：

1. 理解问题
2. 判断是否要用工具
3. 决定调用哪个工具、如何调用

这让系统从“会说话”升级成“会行动”。

## 0. 运行前提醒

这个脚本除了 LangChain 基础依赖外，通常还需要确认这些组件可用：

- `wikipedia`
- `langchain-experimental`

如果你后续把视频里的搜索示例也补回来，往往还会涉及 `duckduckgo-search`。

## 1. 当前项目里的 agent 示例包含什么

`08-agent.py` 其实放了三类 agent 能力：

1. 使用 LangChain 内置工具的通用 agent
2. 使用 Python REPL 的代码执行型 agent
3. 带自定义 `time` 工具的 agent

这三部分组合起来，基本把课程里最重要的 agent 思路都覆盖到了。

## 2. 第一部分：内置工具 agent

### 加载工具

```python
tools = load_tools(["llm-math", "wikipedia"], llm=model)
```

这里和视频有一个小差异：

- 视频常演示 `duckduckgo-search` + `wikipedia`
- 当前项目实际加载的是 `llm-math` + `wikipedia`

这样做的好处是：

- `llm-math` 很容易观察“模型是否知道自己该把计算交给工具”
- `wikipedia` 很适合知识查询类问题

### 初始化 agent

```python
agent = initialize_agent(
    tools,
    model,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
)
```

这里最值得你注意的是三个参数：

- `CHAT_ZERO_SHOT_REACT_DESCRIPTION`
- `handle_parsing_errors=True`
- `verbose=True`

#### `CHAT_ZERO_SHOT_REACT_DESCRIPTION`

它代表一种经典 agent 提示策略：

- 让模型先思考
- 再决定动作
- 再根据工具返回结果继续推理

#### `handle_parsing_errors=True`

agent 的核心风险之一是：模型输出的 action 格式不稳定。

这个参数的作用是：

- 当输出没按预期格式来时
- 不要立刻整条链崩掉
- 尝试继续恢复执行

#### `verbose=True`

学习 agent 阶段强烈建议打开，因为你最想看到的就是：

- 模型做了什么判断
- 选了哪个工具
- 工具返回了什么

## 3. 第二部分：Python Agent

```python
python_agent = create_python_agent(
    llm=model,
    tool=PythonREPLTool(),
    verbose=True,
)
```

这一部分体现了 agent 的另一个重要价值：

> 让模型把不擅长纯脑算的事情，交给外部执行环境。

示例里让 agent 去排序客户名单，这类任务如果只靠模型直接输出，很容易：

- 排错
- 漏项
- 格式不稳定

但如果让它调用 Python REPL，可靠性通常会高很多。

## 4. 第三部分：自定义工具 `time`

```python
@tool
def time(text: str) -> str:
    ...
```

这一节最值得反复体会的是 docstring：

```python
"""Returns todays date ...
The input should always be an empty string ..."""
```

为什么文档字符串这么重要？

因为对 agent 来说，工具说明本身就是“使用说明书”。模型会根据这段描述决定：

- 什么时候该用这个工具
- 输入参数应该怎么传

所以写工具给 agent 用时，docstring 不是装饰，而是 prompt 的一部分。

## 5. 再创建一个带自定义工具的 agent

```python
agent = initialize_agent(
    tools + [time],
    model,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
)
```

然后提问：

```python
"whats the date today?"
```

这时你真正想观察的不是最后那句自然语言回答，而是中间过程：

1. 模型是否识别出“日期问题应该调用工具”
2. 是否正确调用了 `time("")`
3. 是否把工具观察结果转成最终回答

## 6. 当前项目和视频版本的差异

### 差异 1：项目使用 `langchain_classic.agents`

这意味着课程示例仍然沿用经典 agent API，而不是最新官方抽象。

### 差异 2：项目额外补了 Python Agent

视频重点更偏外部搜索工具，当前项目把“用 Python 解决确定性问题”也补上了，这非常实用。

### 差异 3：项目保留了一个异常兜底

```python
try:
    result = agent.invoke({"input": "whats the date today?"})
except:
    print("exception on external access")
```

这提醒你：agent 很强，但也更容易受外部依赖影响。

## 7. 这节最容易踩的坑

### 坑 1：以为 agent 一定比 chain 强

不是。agent 更灵活，但也更复杂、更难控。能用固定链解决的问题，不一定要上 agent。

### 坑 2：工具说明写得太模糊

如果 docstring 不明确，模型就可能：

- 不会用
- 乱用
- 参数传错

### 坑 3：把 agent 当成“永远正确的自动驾驶”

agent 非常依赖：

- 模型推理能力
- 工具返回质量
- 输出解析稳定性

它比普通链更强，也比普通链更脆弱。

## 8. 这一节真正该学到什么

学 agent，不是为了记住 `initialize_agent(...)` 的参数，而是要形成一个心智模型：

1. LLM 负责推理和决策
2. Tool 负责外部能力
3. Agent 框架负责把“思考 -> 动作 -> 观察 -> 最终回答”串起来

## 9. 建议你自己动手改的练习

1. 仿照 `time` 再写一个 `weekday` 工具，让 agent 能回答“今天星期几”。
2. 给 `wikipedia` 类问题和数学类问题各加几组输入，观察 agent 选工具是否稳定。
3. 故意把 `time` 的 docstring 写模糊，再比较 agent 调用表现。

## 10. 学完这节你应该能回答

1. 为什么 agent 可以被理解成“LLM + 工具选择能力”？
2. `CHAT_ZERO_SHOT_REACT_DESCRIPTION` 想解决的核心问题是什么？
3. 为什么自定义工具的 docstring 会直接影响 agent 效果？
4. 什么情况下你应该优先用 chain，而不是 agent？
