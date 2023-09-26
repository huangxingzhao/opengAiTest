# -*- coding:utf-8 -*-
import json

import openai, os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

openai.api_key = "sk-bneyQXujVwm4zZKX1f6e77228a344e3d826a8fD5F431Dc13"
os.environ['OPENAI_API_KEY'] = 'sk-bneyQXujVwm4zZKX1f6e77228a344e3d826a8fD5F431Dc13'
os.environ['OPENAI_API_BASE'] = 'https://api.rcouyi.com/v1'

from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

def recommend_product(input: str) -> str:
    return "红色连衣裙"


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import SpacyTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import TextLoader

llm = OpenAI(temperature=0)
loader = TextLoader('./data/ecommerce_faq.txt')
documents = loader.load()
text_splitter = SpacyTextSplitter(chunk_size=256, pipeline="zh_core_web_sm")
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)

faq_chain = VectorDBQA.from_chain_type(llm=OpenAI(temperature=0), vectorstore=docsearch, verbose=True)


from langchain.agents import tool

@tool("FAQ")
def faq(intput: str) -> str:
    """"useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."""
    return faq_chain.run(intput)


ORDER_1 = "20230101ABC"
ORDER_2 = "20230101EFG"

ORDER_1_DETAIL = {
    "order_number": ORDER_1,
    "status": "已发货",
    "shipping_date" : "2023-01-03",
    "estimated_delivered_date": "2023-01-05",
}

ORDER_2_DETAIL = {
    "order_number": ORDER_2,
    "status": "未发货",
    "shipping_date" : None,
    "estimated_delivered_date": None,
}

import re

@tool("Search Order",return_direct=True)
def search_order(input:str)->str:
    """一个帮助用户查询最新订单状态的工具，并且能处理以下情况：
    1. 在用户没有输入订单号的时候，会询问用户订单号
    2. 在用户输入的订单号查询不到的时候，会让用户二次确认订单号是否正确"""
    pattern = r"\d+[A-Z]+"
    match = re.search(pattern, input)

    order_number = input
    if match:
        order_number = match.group(0)
    else:
        return "请问您的订单号是多少？"
    if order_number == ORDER_1:
        return json.dumps(ORDER_1_DETAIL)
    elif order_number == ORDER_2:
        return json.dumps(ORDER_2_DETAIL)
    else:
        return f"对不起，根据{input}没有找到您的订单"

tools = [
    search_order,
    Tool(name="Recommend Product", func=recommend_product,
         description="useful for when you need to answer questions about product recommendations"
    ),
    faq
]
chatllm=ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(tools, chatllm, agent="zero-shot-react-description", memory=memory,verbose=True)

question = "我有一张订单，一直没有收到，能麻烦帮我查一下吗？"
answer = agent.run(question)
print(answer)

question2 = "我的订单号是20230101ABC"
answer2 = agent.run(question2)
print(answer2)