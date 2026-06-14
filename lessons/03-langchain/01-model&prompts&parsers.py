import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_qwq import ChatQwQ
from langchain_core.prompts import ChatPromptTemplate

_ = load_dotenv(find_dotenv())  # 从 .env 读取 API Key 等配置

temperature = 0

customer_email = """
    Arrr, I be fuming that me blender lid \
    flew off and splattered me kitchen walls \
    with smoothie! And to make matters worse,\
    the warranty don't cover the cost of \
    cleaning up me kitchen. I need yer help \
    right now, matey!
"""

style = """American English \
    in a calm and respectful tone
"""

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{customer_email}```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)  # Prompt（提示词模板）：把 customer_email、style 填进固定模板，生成发给模型的消息

print(prompt_template.messages[0].prompt.input_variables)

customer_style = """American English \
in a calm and respectful tone
"""

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

customer_messages = prompt_template.format_messages(
    style=customer_style, customer_email=customer_email
)

print(customer_messages)

# chat = ChatQwQ(model=model, temperature=temperature)
model = ChatQwQ(model="qwen3.7-plus",temperature=temperature)
response = model.invoke(customer_messages) # 把提示词发给模型，拿到回复
print(response.content)