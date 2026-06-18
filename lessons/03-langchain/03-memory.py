import datetime
import warnings

from dotenv import load_dotenv, find_dotenv
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_qwq import ChatQwQ

_ = load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")

temperature = 0
model = ChatQwQ(model="qwen3.7-plus", temperature=temperature)

# 方式1：ConversationChain 自动管理对话记忆
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=model,
    memory=memory,
    verbose=True,
)

print("=== 多轮对话（模型应记住名字）===")
conversation.predict(input="Hi, my name is Andrew")
conversation.predict(input="What is 1+1?")
conversation.predict(input="What is my name?")

print("\n=== memory.buffer（当前记忆中的对话文本）===")
print(memory.buffer)

print("\n=== load_memory_variables（以变量形式读取记忆）===")
print(memory.load_memory_variables({}))

# 方式2：手动往 memory 里写入上下文
print("\n=== 手动 save_context ===")
memory = ConversationBufferMemory()
memory.save_context({"input": "Hi"}, {"output": "What's up"})
print(memory.buffer)

memory.save_context(
    {"input": "Not much, just hanging"},
    {"output": "Cool"},
)
print(memory.buffer)
print(memory.load_memory_variables({}))
