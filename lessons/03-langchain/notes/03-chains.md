# Chains：把多个 LLM 步骤稳定地串起来

这一节对应：

- `04-chains.py`
- `05-routerChain.py`

课程字幕里把 chain 称为 LangChain 最重要的 building block，这个说法非常准确。因为前面几节你学到的 Prompt、Model、Parser、Memory，最后都要通过“链”来组合成真正的流程。

## 1. 先把 chain 想成什么

一个 chain 可以先粗暴理解成：

> 输入经过一套固定步骤处理，产生输出，并且这套步骤可以重复调用。

在当前项目里，chain 最常见的形态有三种：

1. 一个 Prompt + 一个模型
2. 多个子链按顺序串起来
3. 先路由，再决定走哪个子链

## 2. `04-chains.py`：从单链到顺序链

这个文件使用 `Data.csv` 中的商品和评论数据，分别演示：

- `LLMChain`
- `SequentialChain`

## 3. `LLMChain` 是什么

最基础的写法是：

```python
prompt = ChatPromptTemplate.from_template(
    "基于{product}生成一个最适合用来描述的公司的名字？"
)
chain = LLMChain(llm=model, prompt=prompt)
```

你可以把它看成“一个可复用的提示词函数”：

- 输入：`product`
- 处理：把变量填进 prompt
- 输出：模型生成的公司名

虽然它看起来很简单，但后面的顺序链、路由链，本质上都是在拼这些基础单元。

## 4. 为什么项目里直接用了 `SequentialChain`

视频会先讲 `SimpleSequentialChain`，再讲 `SequentialChain`。当前项目没有单独保留 `SimpleSequentialChain` 示例，而是直接进入更通用的 `SequentialChain`。

这样做反而更适合实战，因为真实业务经常有：

- 多输入
- 多输出
- 中间变量复用

而不只是“一进一出”的最简流程。

## 5. `SequentialChain` 在这个文件里怎么串的

代码一共定义了 4 个子链：

1. 把评论翻译成英文
2. 把英文评论总结成一句话
3. 判断原评论是什么语言
4. 根据 summary 和 language 写一条后续回复

最关键的不是 Prompt 本身，而是这些 `output_key`：

```python
output_key="English_Review"
output_key="summary"
output_key="language"
output_key="followup_message"
```

然后在总链里统一声明：

```python
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["language", "English_Review", "summary", "followup_message"],
    verbose=True,
)
```

## 6. 学 `SequentialChain` 时最应该盯住什么

### 盯住变量名传递

这一节最大的坑几乎都来自变量名不一致。

例如：

- 第一条链输出 `English_Review`
- 第二条链输入也必须叫 `English_Review`
- 第三条链输出 `language`
- 第四条链输入必须能拿到 `summary` 和 `language`

只要名字没对齐，链就会报错。

### 盯住“原始输入”和“中间结果”谁在被复用

这套示例里：

- `Review` 被第一条和第三条链同时使用
- `English_Review` 只给第二条链用
- `summary` 和 `language` 在最后一条链汇合

这就是多步工作流的雏形。

## 7. `05-routerChain.py`：让系统自己选子链

如果说顺序链解决的是“固定流程”，那路由链解决的是：

> 输入来了以后，该走哪条流程？

这个文件定义了四种专家角色：

- physics
- math
- History
- computer science

每个角色都有两样东西：

1. 一套专属提示词
2. 一段给路由器看的说明文字

## 8. 路由链的关键组成

### 第一部分：目标链集合

```python
destination_chains[name] = chain
```

这表示每个专家方向都对应一条真正执行回答的 `LLMChain`。

### 第二部分：路由提示词

```python
MULTI_PROMPT_ROUTER_TEMPLATE = """..."""
```

这个大模板的作用是告诉路由器：

- 你有哪些候选专家
- 每个专家擅长什么
- 你需要输出哪个目标名
- 如果都不合适，就返回 `DEFAULT`

### 第三部分：路由输出解析

```python
output_parser=RouterOutputParser()
```

它会把 LLM 输出解析成结构化结果，例如：

- `destination`
- `next_inputs`

这跟第二节的 Output Parser 是同一个思路，只是这次解析的不是业务字段，而是“路由决策”。

### 第四部分：组装 `MultiPromptChain`

```python
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)
```

这时候整条链就变成了：

1. 先让路由模型判断问题类型
2. 再把问题交给对应专家子链
3. 如果不匹配，就走默认链

## 9. 当前项目和视频版本的几个实用差异

### 差异 1：项目把 Router 单独拆成一个完整脚本

这比课程里混在 chain 章节里更好，因为你能更清楚看到：

- 路由器自己也是一条链
- 目标专家链是另一批链
- parser 在这里的作用非常关键

### 差异 2：项目用中文专家设定，更适合本地实验

视频里主要是英文场景，当前项目把专家说明改成了中文提示词，更方便你直接在中文问题上观察路由效果。

### 差异 3：目标名必须精确匹配

这里尤其要注意：

- `prompt_infos` 里的 `name`
- 路由器输出里的 `destination`
- `destination_chains` 的 key

这三者必须一致，否则就会路由失败或落到默认链。

## 10. 这节真正的工程意义

Chains 的价值不只是“能串起来”，更重要的是：

- 让每一步职责清晰
- 让中间变量可检查
- 让复杂流程可复用
- 让调试更容易定位到哪一步出问题

如果没有链，你也能手写这些逻辑，但很快就会陷入：

- prompt 到处散落
- 中间变量命名混乱
- 调用顺序靠人脑记

## 11. 建议你自己动手改的练习

1. 给 `04-chains.py` 再加一条链，让系统在最后顺便输出“评论情感”。
2. 把 `05-routerChain.py` 扩展出一个 `biology` 专家，再问一组生物学问题。
3. 故意把某个 `output_key` 改错，看看报错信息会指向哪里。

## 12. 学完这节你应该能回答

1. `LLMChain` 和单次 `model.invoke(...)` 的差别是什么？
2. `SequentialChain` 为什么比 `SimpleSequentialChain` 更接近真实业务？
3. 路由链为什么也需要 Output Parser？
4. 一个复杂 LangChain 工作流里，变量命名为什么会直接影响正确性？
