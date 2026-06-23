from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_classic.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import CSVLoader
from langchain_classic.chains import RetrievalQA
import os
from pathlib import Path
from dashscope import TextEmbedding
from typing import List
import dashscope
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_qwq import ChatQwQ
from langchain_classic.evaluation.qa import QAGenerateChain, QAEvalChain
from langchain_classic.output_parsers.regex import RegexParser
from langchain_core.globals import set_debug

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # 从 .env 读取 API Key 等配置

# 中文问答生成提示词（配合下方 RegexParser 解析「问题 / 答案」）
QA_GENERATE_PROMPT = PromptTemplate(
    input_variables=["doc"],
    template="""你是一位老师，正在根据文档编写测验题。
请根据以下文档生成一个问题和答案。

格式示例：
<开始文档>
...
<结束文档>
问题：问题内容
答案：答案内容

问题应详细且明确基于文档中的信息。开始！

<开始文档>
{doc}
<结束文档>""",
)

QA_GENERATE_PARSER = RegexParser(
    regex=r"问题[:：]\s*(.*?)\n+答案[:：]\s*(.*)",
    output_keys=["query", "answer"],
)

# 中文评估提示词（评分结果仍输出 CORRECT / INCORRECT，便于框架解析）
QA_EVAL_PROMPT = PromptTemplate(
    input_variables=["query", "result", "answer"],
    template="""你是一位老师，正在批改测验。
你会收到一个问题、学生的答案和标准答案，请将学生答案评为 CORRECT（正确）或 INCORRECT（错误）。

格式示例：
问题：此处是问题
学生答案：此处是学生答案
标准答案：此处是标准答案
评分：CORRECT 或 INCORRECT

仅根据事实准确性评分。忽略标点与措辞差异。学生答案可以比标准答案包含更多信息，只要不包含矛盾陈述。开始！

问题：{query}
学生答案：{result}
标准答案：{answer}
评分：""",
)


class DashScopeEmbeddings(Embeddings):
    """百炼 embedding API，与 ChatQwQ 共用 DASHSCOPE_API_KEY。"""

    def __init__(self, model: str = TextEmbedding.Models.text_embedding_v3):
        dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
        self.model = model

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        resp = TextEmbedding.call(model=self.model, input=texts)
        if resp.status_code != 200:
            raise RuntimeError(resp)
        return [item["embedding"] for item in resp.output["embeddings"]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), 10):  # 百炼单次最多 10 条
            embeddings.extend(self._call_api(texts[i: i + 10]))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._call_api([text])[0]


file = Path(__file__).parent / "OutdoorClothingCatalog_1000.csv"
loader = CSVLoader(file_path=file)
data = loader.load()
embedding = DashScopeEmbeddings()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embedding,
).from_loaders([loader])
model = ChatQwQ(model="qwen3.7-plus", temperature=0)
qa = RetrievalQA.from_chain_type(
    llm=model,  # 指定语言模型
    chain_type="stuff",  # 链类型：将全部检索文档塞入 prompt
    retriever=index.vectorstore.as_retriever(),  # 检索器
    verbose=True,
    chain_type_kwargs={
        "document_separator": "<<<<>>>>>"
    }
)
# print(data[10])
# print(data[11])

examples = [
    {
        "query": "舒适居家套头套装有侧边口袋吗？",
        "answer": "是"
    },
    {
        "query": "Ultra-Lofty 850 弹力连帽羽绒服属于哪个系列？",
        "answer": "DownTek 系列"
    }
]

example_gen_chain = QAGenerateChain(
    llm=model,
    prompt=QA_GENERATE_PROMPT,
    output_parser=QA_GENERATE_PARSER,
)

new_examples = [
    result["qa_pairs"]
    for result in example_gen_chain.apply([{"doc": t} for t in data[:5]])
]
# print(new_examples)

examples += new_examples

# 开启全局 debug：打印更详细的链执行日志（qa 已设 verbose=True，二者可叠加）
set_debug(True)
for i in range(2):
    result = qa.invoke({"query": examples[i]["query"]})
    print(f"问题 {i + 1}：{examples[i]['query']}")
    print(f"回答 {i + 1}：{result['result']}\n")
set_debug(False)

predictions = qa.apply(examples)
eval_chain = QAEvalChain.from_llm(model, prompt=QA_EVAL_PROMPT)
graded_outputs = eval_chain.evaluate(examples, predictions)
for i in range(len(examples)):
    print(f"示例 {i + 1}：")
    print("问题：" + predictions[i]["query"])
    print("标准答案：" + examples[i]["answer"])
    print("模型答案：" + predictions[i]["result"])
    print("评分：" + graded_outputs[i]["results"])
    print()
print(graded_outputs[0])
