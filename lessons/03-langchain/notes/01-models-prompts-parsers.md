# 模型、提示词与结构化输出

这一节对应课程字幕里的“Models / Prompts / Parsers”，但在当前项目里被拆成了两部分：

- `01-model&prompts&parsers.py`：模型调用 + Prompt 模板
- `02-llm生成json.py`：结构化输出解析

也就是说，**项目实现比视频更清晰**：先学“怎么问”，再学“怎么把回答变成结构化数据”。

## 1. 这一节到底在学什么

LangChain 在这一节提供的核心价值只有三件事：

1. 用统一接口包装模型
2. 用模板安全地构造 prompt
3. 把模型输出解析成结构化对象

如果你只记一句话，那就是：

> LangChain 不负责替你“想问题”，它主要负责把“提示词构造、模型调用、结果解析”变成可复用组件。

## 2. `01-model&prompts&parsers.py` 在做什么

这个文件的主线非常简单：把英文文本改写成不同风格。

### 关键步骤

#### 第一步：加载环境变量

```python
_ = load_dotenv(find_dotenv())
```

作用是让 `ChatQwQ` 能拿到 API Key 和 Base URL。

#### 第二步：定义 Prompt 模板

```python
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{customer_email}```"""

prompt_template = ChatPromptTemplate.from_template(template_string)
```

这里最重要的不是翻译任务本身，而是 `{style}` 和 `{customer_email}` 这两个变量。

`ChatPromptTemplate` 的意义在于：

- 模板和变量分离
- 避免手写字符串拼接
- 后续更容易复用、调试、替换变量

#### 第三步：把变量填进模板

```python
customer_messages = prompt_template.format_messages(
    style=customer_style,
    customer_email=customer_email,
)
```

这一行是很多初学者第一次接触 LangChain 时的关键转折点：

- 你最终传给模型的，不一定是一个普通字符串
- LangChain 更喜欢“消息列表”的形式
- `format_messages()` 返回的是可直接喂给聊天模型的消息对象

#### 第四步：调用模型

```python
model = ChatQwQ(model="qwen3.7-plus", temperature=temperature)
response = model.invoke(customer_messages)
```

这里的重点不是 `qwen3.7-plus`，而是 `invoke(...)`。

你可以把它理解成：

- 老思路：`prompt -> API -> string`
- LangChain 思路：`messages -> model.invoke(...) -> AIMessage`

## 3. 当前项目和视频版本的差异

### 差异 1：模型从 OpenAI 换成了通义封装

视频里核心是 `ChatOpenAI`，项目里是 `ChatQwQ`。但真正不变的是：

- 都实现了 LangChain 的聊天模型接口
- 都能吃消息列表
- 都能通过 `invoke` 返回消息对象

所以你应该学接口，而不是学死某个供应商。

### 差异 2：文件名里有 `parsers`，但实际 parser 在下一个文件

这不是错误，而是当前项目为了教学节奏做了拆分：

- `01` 先把 Prompt 和模型吃透
- `02` 再进入结构化输出

### 差异 3：`OpenAI` 和 `os` 在文件里没有真正参与执行

这说明当前文件是从课程原稿改造过来的，保留了一些旧导入。学习时别被这些冗余代码干扰，重点仍然是：

- `ChatPromptTemplate`
- `format_messages`
- `model.invoke`

## 4. `02-llm生成json.py` 为什么更重要

很多同学学 LangChain 时停留在“让模型回答一段话”，但真实项目里更常见的需求是：

- 提取字段
- 生成 JSON
- 生成可校验对象
- 把结果继续交给下游程序

这个文件正是在做这件事。

### 关键结构

#### 用 Pydantic 定义目标输出

```python
class ReviewInfo(BaseModel):
    gift: str
    delivery_days: int
    price_value: List[str]
```

这一步相当于先定义“我希望模型最后长成什么样子”。

#### 用输出解析器约束格式

```python
output_parser = PydanticOutputParser(pydantic_object=ReviewInfo)
```

这意味着模型输出不再只是“看起来像 JSON”就行，而是要能被真正解析成 `ReviewInfo`。

#### 把格式要求塞进 prompt

```python
{format_instructions}
```

这个变量来自：

```python
output_parser.get_format_instructions()
```

也就是说，LangChain 不只是“解析失败后报错”，它还会提前把格式规则告诉模型。

#### 用 LCEL 直接串起来

```python
chain = prompt | model | output_parser
```

这一行很值得反复看，因为它代表了更现代的 LangChain 写法：

- `prompt` 负责生成输入
- `model` 负责生成回答
- `output_parser` 负责把回答转成结构化对象

## 5. 这一节最该记住的两个心智模型

### 心智模型 1：Prompt 模板是“参数化的消息工厂”

它不是简单字符串模板，而是为了输出适合聊天模型的消息对象。

### 心智模型 2：Output Parser 是“LLM 到程序对象”的桥

如果没有 parser：

- 你得到的是一段文字
- 你还得自己 `json.loads`
- 出错时只能手动补救

有了 parser：

- 结果直接落到 `Pydantic` 对象
- 字段类型更清晰
- 后续链路更容易维护

## 6. 真实项目里最容易踩的坑

### 坑 1：变量名和语义不一致

在 `01-model&prompts&parsers.py` 里，模板变量叫 `customer_email`，但第二次调用实际传入的是 `service_reply`。虽然程序能跑，但读代码的人会困惑。

结论：模板变量名尽量写成更通用的 `text`。

### 坑 2：模型没有严格按 JSON 输出

即使有格式说明，模型有时也会输出解释性文字。`PydanticOutputParser` 会帮你更早暴露问题，但你仍要有“提示词不稳导致解析失败”的心理准备。

### 坑 3：把 LangChain 当成“自动变聪明”

LangChain 不会自动提高模型质量。它提升的是：

- 代码可复用性
- 组合能力
- 可维护性

## 7. 建议你自己动手改的练习

1. 把 `01-model&prompts&parsers.py` 的模板变量名从 `customer_email` 改成 `text`，看看代码可读性是不是马上变好。
2. 在 `02-llm生成json.py` 里给 `ReviewInfo` 新增一个 `sentiment` 字段，让模型顺便输出评论情感。
3. 故意把 `format_instructions` 去掉，再比较一次解析稳定性。

## 8. 学完这节你应该能回答

1. `ChatPromptTemplate.from_template(...)` 和手写 f-string 的区别是什么？
2. `format_messages(...)` 为什么比直接拼字符串更适合聊天模型？
3. `prompt | model | output_parser` 这条链里每一段分别负责什么？
4. 为什么真实项目里“结构化输出”通常比“自由文本输出”更重要？
