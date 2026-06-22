import os
from dotenv import load_dotenv, find_dotenv
import warnings
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAI
from langchain_classic.document_loaders import CSVLoader
from langchain_classic.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown

warnings.filterwarnings('ignore')
_ = load_dotenv(find_dotenv())  # read local .env file
