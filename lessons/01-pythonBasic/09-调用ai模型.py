import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv('.env', override=True) # 加载.env文件,override=True表示如果.env文件中有变量,则覆盖原来的变量

api_key = os.getenv("api_key")
client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)