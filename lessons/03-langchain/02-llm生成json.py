import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_qwq import ChatQwQ
from langchain_core.prompts import ChatPromptTemplate

_ = load_dotenv(find_dotenv())  # 从 .env 读取 API Key 等配置

temperature = 1



customer_review = """
这款吹叶机非常厉害，有四种模式：烛光模式、轻风模式、城市强风模式和龙卷风模式。
两天内就送到了，正好赶上我妻子的生日礼物。
我觉得我妻子非常喜欢，简直说不出话来。
到目前为止，我一直是唯一使用它的人，而且我每天早上都会用它来清理草坪上的落叶。
它比市面上其他吹叶机稍贵一些，但我认为额外的功能值得。
"""

review_template = """
请从以下文本中提取以下信息：
gift：这件物品是作为赠送给他人的礼物购买的吗？
如果为真，请回答“yes”；如果为假，请回答“no”；如果不确定，请回答“unknown”。
price_value：产品到达需要多少天？如果无法找到此信息，则输出 -1。
price_value：提取所有关于价值或价格的句子，并以逗号分隔的Python列表形式输出。
以以下键值格式化输出为 JSON：
gift
delivery_days
price_value
内容: {text}
"""

prompt_template = ChatPromptTemplate.from_template(review_template).format_messages(text=customer_review)  # Prompt
model = ChatQwQ(model="qwen3.7-plus",temperature=temperature)
response = model.invoke(prompt_template)
"""
{
    "gift": "yes",
    "delivery_days": 2,
    "price_value": ["它比市面上其他吹叶机稍贵一些，但我认为额外的功能值得。"]
}
"""
print(response.content)
type(response.content)  # str ，实际上是一个字符串不是json

from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class ReviewInfo(BaseModel):
    gift: str = Field(description="这件物品是作为赠送给他人的礼物购买的吗？")
    delivery_days: int = Field(description="产品到达需要多少天？")
    price_value: List[str] = Field(
        description="提取所有关于价值或价格的句子，并以逗号分隔的Python列表形式输出。"
    )


output_parser = PydanticOutputParser(pydantic_object=ReviewInfo)  # 创建结构化输出解析器
format_instructions = output_parser.get_format_instructions()
print(format_instructions)