import json
from typing import List

from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwQ
from pydantic import BaseModel, Field

_ = load_dotenv(find_dotenv())

temperature = 0

customer_review = """
这款吹叶机非常厉害，有四种模式：烛光模式、轻风模式、城市强风模式和龙卷风模式。
两天内就送到了，正好赶上我妻子的生日礼物。
我觉得我妻子非常喜欢，简直说不出话来。
到目前为止，我一直是唯一使用它的人，而且我每天早上都会用它来清理草坪上的落叶。
它比市面上其他吹叶机稍贵一些，但我认为额外的功能值得。
"""


class ReviewInfo(BaseModel):
    gift: str = Field(description="这件物品是作为赠送给他人的礼物购买的吗？")
    delivery_days: int = Field(description="产品到达需要多少天？")
    price_value: List[str] = Field(
        description="提取所有关于价值或价格的句子，并以逗号分隔的 Python 列表形式输出。"
    )


output_parser = PydanticOutputParser(pydantic_object=ReviewInfo)

review_template = """
请从以下文本中提取信息，并严格按指定格式输出。

gift：这件物品是作为赠送给他人的礼物购买的吗？
如果为真，请回答 yes；如果为假，请回答 no；如果不确定，请回答 unknown。

delivery_days：产品到达需要多少天？如果无法找到此信息，则输出 -1。

price_value：提取所有关于价值或价格的句子，以 Python 列表形式输出。

{format_instructions}

内容: {text}
"""

prompt = ChatPromptTemplate.from_template(review_template)
model = ChatQwQ(model="qwen3.7-plus", temperature=temperature)

chain = prompt | model | output_parser

result = chain.invoke({
    "text": customer_review,
    "format_instructions": output_parser.get_format_instructions(),
})

print("结构化对象:", result)
print("类型:", type(result))
print("gift:", result.gift)
print("delivery_days:", result.delivery_days)
print("price_value:", result.price_value)
print("JSON 字符串:")
print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
