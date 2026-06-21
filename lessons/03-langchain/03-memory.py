"""
LangChain Memory（记忆）示例

大模型本身是无状态的：每次 invoke 都是独立请求，不会记住上一轮说了什么。
Memory 的作用是把历史对话保存下来，在下一轮请求时一并塞进 prompt，让模型"看起来"有记忆。

本文件演示 ConversationBufferMemory：把完整对话原文存入 buffer，不做截断或摘要。
"""

from dotenv import load_dotenv, find_dotenv
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_qwq import ChatQwQ
import warnings

_ = load_dotenv(find_dotenv())  # 从 .env 读取 API Key 等配置
warnings.filterwarnings("ignore")

temperature = 0
model = ChatQwQ(model="qwen3.7-plus", temperature=temperature)

# ---------------------------------------------------------------------------
# 方式 1：ConversationChain 自动管理对话记忆
# ---------------------------------------------------------------------------
# ConversationBufferMemory：最简单的记忆类型，对话缓冲区内存。完整保存每一轮 input / output
memory = ConversationBufferMemory()

# ConversationChain = LLM + Memory + 默认对话 prompt
# 每次 predict 时：读取 memory → 拼进 prompt → 调用模型 → 把本轮问答写回 memory
conversation = ConversationChain(
    llm=model,
    memory=memory,
    verbose=True,  # 打印完整 prompt，方便观察 memory 如何被注入
)

print("=== 多轮对话（模型应记住名字）===")
conversation.predict(input="Hi, my name is Andrew")  # 第 1 轮：自我介绍
conversation.predict(input="What is 1+1?")           # 第 2 轮：无关问题，但 memory 仍在累积
conversation.predict(input="What is my name?")       # 第 3 轮：测试模型能否从历史中找回名字

# memory.buffer：人类可读的完整对话文本（Human: ... / AI: ... 交替拼接）
print("\n=== memory.buffer（当前记忆中的对话文本）===")
print(memory.buffer)

# load_memory_variables：以字典形式读取，键名默认为 "history"，供 prompt 模板占位符使用
print("\n=== load_memory_variables（以变量形式读取记忆）===")
print(memory.load_memory_variables({}))

# ---------------------------------------------------------------------------
# 方式 2：手动往 memory 里写入上下文
# ---------------------------------------------------------------------------
# 适用场景：从数据库、日志等外部来源恢复历史，而非通过 ConversationChain 逐轮对话
print("\n=== 手动 save_context ===")
memory = ConversationBufferMemory()
# save_context 接收一对 input / output，格式与 ConversationChain 内部写入的一致
memory.save_context({"input": "Hi"}, {"output": "What's up"})
print(memory.buffer)

memory.save_context(
    {"input": "Not much, just hanging"}, 
    {"output": "Cool"},
)
print(memory.buffer)
print(memory.load_memory_variables({}))
