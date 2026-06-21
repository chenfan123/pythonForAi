from dotenv import load_dotenv, find_dotenv
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationTokenBufferMemory,
)
from langchain_openai import ChatOpenAI
from langchain_qwq import ChatQwQ
import warnings

_ = load_dotenv(find_dotenv())  # 从 .env 读取 API Key 等配置
warnings.filterwarnings("ignore")

temperature = 0
model = ChatQwQ(model="qwen3.7-plus", temperature=temperature)

# 方式 1：按轮数截断，只保留最近 k 轮对话
window_memory = ConversationBufferWindowMemory(k=2)
window_memory.save_context({"input": "Hi, my name is Andrew"}, {"output": "What's up"})
window_memory.save_context({"input": "What's up"}, {"output": "I'm good, thank you"})
window_memory.save_context({"input": "how are you"}, {"output": "I'm fine,thank you"})
print(window_memory.load_memory_variables({}))

# 方式 2：按 token 数截断，超出 max_token_limit 时丢弃最早的消息
# ConversationTokenBufferMemory 需要 llm.get_num_tokens_from_messages()，
# ChatQwQ 未实现该方法，故用 gpt-3.5-turbo 仅作本地 token 计数（不发起 API 调用）
token_counter = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
token_memory = ConversationTokenBufferMemory(llm=token_counter, max_token_limit=100)
token_memory.save_context({"input": "Hi, my name is Andrew"}, {"output": "What's up"})
token_memory.save_context({"input": "What's up111"}, {"output": "I'm good, thank you222"})
token_memory.save_context({"input": "how are you"}, {"output": "I'm fine,thank you"})
print(token_memory.load_memory_variables({}))

# 方式 3：用 LLM 持续生成对话摘要，history 返回的是摘要文本而非完整对话
schedule = """There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."""

summary_memory = ConversationSummaryMemory(llm=model)  # llm 必须是模型实例，不能传字符串
summary_memory.save_context({"input": "Hello"}, {"output": "What's up"})
summary_memory.save_context(
    {"input": "Not much, just hanging"},
    {"output": "Cool"},
)
summary_memory.save_context(
    {"input": "What is on the schedule today?"},
    {"output": schedule},
)

conversation = ConversationChain(
    llm=model,  # 同样需要传入模型实例
    memory=summary_memory,
    verbose=True,
)
conversation.predict(input="What would be a good demo to show?")
print(summary_memory.load_memory_variables({}))
