import os
from pathlib import Path
from typing import List

import dashscope
from dashscope import TextEmbedding
from dotenv import load_dotenv, find_dotenv
import warnings
from langchain_community.document_loaders import CSVLoader  # 文档加载器，用于加载 CSV 文件
from langchain_classic.indexes import VectorstoreIndexCreator  # 向量索引器，用于创建向量索引
from langchain_community.vectorstores import DocArrayInMemorySearch  # 向量存储，用于存储文档的向量表示
from langchain_core.embeddings import Embeddings
from langchain_qwq import ChatQwQ
from langchain_classic.chains import RetrievalQA
warnings.filterwarnings("ignore")
_ = load_dotenv(find_dotenv())  # 从 .env 读取 API Key 等配置


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
            embeddings.extend(self._call_api(texts[i : i + 10]))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._call_api([text])[0]


file = Path(__file__).parent / "OutdoorClothingCatalog_1000.csv"
loader = CSVLoader(file_path=file)
docs = loader.load()
# print(docs[0])

embedding = DashScopeEmbeddings()
embed = embedding.embed_query("Hi my name is Harrison")
# print(len(embed))  # 1024
# print(embed[:5]) # 前5个元素[-0.015097519382834435, -0.018166523426771164, -0.06918129324913025, -0.04775766283273697, -0.05385607108473778]
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embedding
)  # 接受一个文档列表作为输入参数，以及一个embedding模型作为输入参数。然后会创建一个向量存储
query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)
# print(len(docs)) # 4份文档
# print(docs[0].page_content[:100])  # : 28 name: Performance Plus 梭织衬衫description: 这款透气夏季衬衫非常适合徒步或旅行，拥有棉质的外观和触感——但充满了高性能。尺寸版型 微修身：柔和地勾勒身
# print(docs[1])

retriever = db.as_retriever() # 创建retriever，是一种通用的接口可以用于检索向量数据库中的文档
model = ChatQwQ(model="qwen3.7-plus", temperature=0)
qdocs = "".join(doc.page_content for doc in docs)
response = model.invoke(
    f"{qdocs}\nQuestion: Please list all your shirts with sun protection "
    "in a table in markdown and summarize each one."
)
# print(f"response11: {response.content}")

qa_stuff = RetrievalQA.from_chain_type(
    llm=model, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."
response = qa_stuff.run(query)
print(f"response22: {response}")
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embedding,
).from_loaders([loader])

query = "请以 Markdown 表格形式列出所有具有防晒功能的衬衫，并对每款进行简要总结。"
result = index.query(query, llm=model)
# print(result)
