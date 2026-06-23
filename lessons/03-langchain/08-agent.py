from dotenv import load_dotenv, find_dotenv
import warnings

_ = load_dotenv(find_dotenv())  # 从 .env 读取 API Key 等配置
warnings.filterwarnings("ignore")

from langchain_classic.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.globals import set_debug
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_qwq import ChatQwQ
from langchain_core.tools import tool
from datetime import date

temperature = 0
model = ChatQwQ(model="qwen3.7-plus", temperature=temperature)

tools = load_tools(["llm-math", "wikipedia"], llm=model)
agent = initialize_agent(
    tools,
    model,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
)

result = agent.invoke({"input": "What is the 25% of 300?"})
print(result)

question = (
    "Tom M. Mitchell is an American computer scientist "
    "and the Founders University Professor at Carnegie Mellon University (CMU) "
    "what book did he write?"
)
result = agent.invoke({"input": question})
print(result)

python_agent = create_python_agent(
    llm=model,
    tool=PythonREPLTool(),
    verbose=True,
)

customer_list = [
    ["Harrison", "Chase"],
    ["Lang", "Chain"],
    ["Dolly", "Too"],
    ["Elle", "Elem"],
    ["Geoff", "Fusion"],
    ["Trance", "Former"],
    ["Jen", "Ayai"],
]

sort_prompt = (
    "Sort these customers by last name and then first name "
    f"and print the output: {customer_list}"
)
python_agent.invoke({"input": sort_prompt})

set_debug(True)
python_agent.invoke({"input": sort_prompt})
set_debug(False)

@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())

agent= initialize_agent(
    tools + [time], 
    model, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True)
    
try:
    result = agent.invoke({"input": "whats the date today?"})
except: 
    print("exception on external access")