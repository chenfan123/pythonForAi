from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


class Answer(BaseModel):
    summary: str
    confidence: float


agent = create_agent(model="ollama:qwen2.5:7b", tools=[
                     search], system_prompt="你是一个智能助手", response_format=Answer)

# 调用聊天模型方式1（invoke）
# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "你好，你是谁"}]})

# # 取回答文本
# answer_text = result["messages"][-1].content
# print("回答:", answer_text)
# # 查看消息类型
# for msg in result["messages"]:
#     print(type(msg).__name__, "->", msg.content)

# 调用方式2（stream）
# stream 返回的是 dict 或 (message_chunk, metadata) 元组，没有 .text 属性
# 使用 stream_mode="messages" 可逐 token 输出模型回复
# for message_chunk, _metadata in agent.stream(
#     input={"messages": [
#         {"role": "user", "content": "请思考后帮我逐步列出做一道红烧羊肉的具体流程?"}]},
#     stream_mode="messages",
# ):
#     # print(_metadata)
#     if message_chunk.content:
#         print(message_chunk.content, end="", flush=True)

# print()
inputs = [
    {"messages": [
        {"role": "user", "content": "Why do parrots have colorful feathers?"}]},
    {"messages": [{"role": "user", "content": "How do airplanes fly?"}]},
    {"messages": [{"role": "user", "content": "What is quantum computing?"}]}
]
# 调用方式3（batch）
# batch 不要加 stream_mode="messages"，否则会返回大量流式 chunk
responses = agent.batch(inputs=inputs)

for index, response in enumerate(responses, start=1):
    question = inputs[index - 1]["messages"][0]["content"]
    answer = response["messages"][-1].content
    print(f"问题 {index}: {question}")
    print(f"回答: {answer}\n")
