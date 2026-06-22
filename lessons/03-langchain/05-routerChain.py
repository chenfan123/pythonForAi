"""
LangChain Router Chain（路由链）示例

根据用户问题的类型，自动选择最合适的「专家 Chain」来回答。
整体由路由器 + 多个目标 Chain + 默认 Chain 组成。
"""

from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import warnings
import pandas as pd
from langchain_qwq import ChatQwQ
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# MultiPromptChain：由路由器根据输入选择最合适的专家 Chain（每条 Chain 对应一套提示词），
# 无匹配时走默认 Chain；也可在路由时改写输入。
from langchain_classic.chains.router import MultiPromptChain
# LLMRouterChain 利用语言模型本身在不同的子chain中来进行路由处理；
# RouterOutputParser：解析路由器返回的 JSON 输出，提取 destination 和 next_inputs。
from langchain_classic.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain_classic.chains import LLMChain

warnings.filterwarnings('ignore')
_ = load_dotenv(find_dotenv())  # 从 .env 读取 API Key 等配置

# ---------------------------------------------------------------------------
# 数据加载（本示例主要演示路由，Data.csv 可选作后续扩展）
# ---------------------------------------------------------------------------
data_path = Path(__file__).resolve().parent / "Data.csv"
df = pd.read_csv(data_path)
df.head()

# ---------------------------------------------------------------------------
# 第一步：为不同领域定义专家提示词模板
# 每个模板描述一个「角色」，占位符 {input} 接收用户问题
# ---------------------------------------------------------------------------
physics_template = """你是一位非常聪明的物理学教授。\
你擅长用简洁易懂的方式回答物理问题。\
当你不知道答案时，你会坦诚承认不知道。

问题如下：
{input}"""


math_template = """你是一位非常优秀的数学家。\
你擅长回答数学问题。\
你的厉害之处在于能把难题拆成若干小部分，\
分别求解后再组合起来，回答整体问题。

问题如下：
{input}"""

history_template = """你是一位非常优秀的历史学家。\
你对不同历史时期的人物、事件和背景都有深入了解。\
你善于思考、反思、辩论、讨论和评价过去。\
你尊重历史证据，并能运用证据来支撑你的解释和判断。

问题如下：
{input}"""


computerscience_template = """你是一位成功的计算机科学家。\
你富有创造力，善于协作，思维前瞻，自信且具备很强的问题解决能力。\
你理解各种理论与算法，沟通能力也很出色。\
你擅长回答编程相关问题。\
你的优势在于：能把解决方案描述成机器易于执行的步骤，\
并且能在时间复杂度和空间复杂度之间做出合理权衡。

问题如下：
{input}"""

# name：路由目标标识，须与路由器返回的 destination 一致
# description：供路由器判断「什么问题该走这条链」
prompt_infos = [
    {
        "name": "physics",
        "description": "适合回答物理相关问题",
        "prompt_template": physics_template
    },
    {
        "name": "math",
        "description": "适合回答数学相关问题",
        "prompt_template": math_template
    },
    {
        "name": "History",
        "description": "适合回答历史相关问题",
        "prompt_template": history_template
    },
    {
        "name": "computer science",
        "description": "适合回答计算机科学相关问题",
        "prompt_template": computerscience_template
    }
]

temperature = 0
model = ChatQwQ(model="qwen3.7-plus", temperature=temperature)

# ---------------------------------------------------------------------------
# 第二步：为每个领域创建 destination_chain（目标链）
# 路由器选中某个 name 后，会调用对应的 LLMChain
# ---------------------------------------------------------------------------
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=model, prompt=prompt)
    destination_chains[name] = chain

# 拼成「名称: 描述」列表，注入路由提示词，供 LLM 选择目标
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

# 默认链：当路由器返回 destination="DEFAULT" 时使用（无匹配的专家）
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=model, prompt=default_prompt)

# ---------------------------------------------------------------------------
# 第三步：配置路由器提示词
# 路由器 LLM 根据用户 input 输出 JSON：destination + next_inputs
# ---------------------------------------------------------------------------
MULTI_PROMPT_ROUTER_TEMPLATE = """给定一条发送给语言模型的原始输入，\
请选择最适合该输入的提示词。\
你会看到可用提示词的名称，以及每个提示词最适合处理的内容描述。\
如果你认为修改原始输入能让语言模型给出更好的回答，也可以对输入进行改写。

<< 格式 >>
返回一个 markdown 代码块，其中 JSON 对象格式如下：
```json
{{{{
    "destination": string \\ "DEFAULT" 或 {destinations} 中的某个提示词名称
    "next_inputs": string \\ 可能是经过修改的原始输入
}}}}
```

请记住："destination" 的值必须与下方列出的候选提示词之一匹配。\
如果没有合适的提示词，请将 "destination" 设置为 "DEFAULT"。
请记住：如果你认为不需要修改输入，\
"next_inputs" 可以直接使用原始输入。

<< 候选提示词 >>
{destinations}

<< 输入 >>
{{input}}

<< 输出（记得包含 ```json）>>"""

# 将候选链列表填入模板（运行时只剩 {input} 一个变量）
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),  # 把 LLM 返回的 JSON 解析成路由结果
)

router_chain = LLMRouterChain.from_llm(model, router_prompt)

# ---------------------------------------------------------------------------
# 第四步：组装 MultiPromptChain
# 流程：input → router_chain 选目标 → 对应 destination_chain 处理
# ---------------------------------------------------------------------------
chain = MultiPromptChain(router_chain=router_chain,
                         destination_chains=destination_chains,
                         default_chain=default_chain, verbose=True  # verbose=True 时会打印路由决策过程
                         )

# ---------------------------------------------------------------------------
# 测试：不同领域的问题应路由到不同专家 Chain
# verbose=True 时会打印路由决策过程
# ---------------------------------------------------------------------------
print(chain.run("What is black body radiation?"))  # 预期 → physics

print(chain.run("what is 2 + 2"))  # 预期 → math

# 预期 → biology 无匹配时可能走 DEFAULT 或相近领域
print(chain.run("Why does every cell in our body contain DNA?"))
