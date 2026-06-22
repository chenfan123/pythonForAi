from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import warnings
import pandas as pd
from langchain_qwq import ChatQwQ
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import LLMChain, SequentialChain

warnings.filterwarnings('ignore')
_ = load_dotenv(find_dotenv())  # read local .env file

data_path = Path(__file__).resolve().parent / "Data.csv"
df = pd.read_csv(data_path)
df.head()

temperature = 0
model = ChatQwQ(model="qwen3.7-plus", temperature=temperature)

prompt = ChatPromptTemplate.from_template(
    "基于{product}生成一个最适合用来描述的公司的名字？"
)
chain = LLMChain(llm=model, prompt=prompt)

product = df.iloc[0]["产品"]
# print(chain.run(product=product)) # 暂时注释掉，下面看SequentialChain的用法

# prompt template 1: translate to english
first_prompt = ChatPromptTemplate.from_template(
    "将下述评论翻译成英文:"
    "\n\n{Review}"
)
# chain 1: input= Review and output= English_Review
chain_one = LLMChain(llm=model, prompt=first_prompt,
                     output_key="English_Review"
                     )

second_prompt = ChatPromptTemplate.from_template(
    "总结以下评论为一句话："
    "\n\n{English_Review}"
)
# chain 2: input= English_Review and output= summary
chain_two = LLMChain(llm=model, prompt=second_prompt,
                     output_key="summary"
                     )

# prompt template 3: translate to english
third_prompt = ChatPromptTemplate.from_template(
    "以下评论是什么语言的：\n\n{Review}"
)
# chain 3: input= Review and output= language
chain_three = LLMChain(llm=model, prompt=third_prompt,
                       output_key="language"
                       )


# prompt template 4: follow up message
fourth_prompt = ChatPromptTemplate.from_template(
    "写一条后续回复给以下总结，使用指定语言："
    "\n\n总结: {summary}\n\n语言: {language}"
)

# chain 4: input= summary, language and output= followup_message
chain_four = LLMChain(llm=model, prompt=fourth_prompt,
                      output_key="followup_message"
                      )

# overall_chain: input= Review
# and output= English_Review,summary, followup_message
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["language", "English_Review",
                      "summary", "followup_message"],
    verbose=True,
)
review = df.iloc[5]["评论"]
result = overall_chain({"Review": review})
print(result)
