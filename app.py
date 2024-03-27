from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv
import os
from typing import List
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LCPC
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain import hub



llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0)
load_dotenv()

def query_pinecone():
    """Look up for product managment info."""
    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    # Access the API key using os.getenv
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    # Connect to Pinecone and specify your Pinecone API key
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = 'meshlennysnews'
    index = pc.Index(index_name)
    text_field = "text"
    # initialize the vector store object
    vectorstore = LCPC(index, embed_model.embed_query, text_field)
    retriever = vectorstore.as_retriever()
    
    return retriever


PM_retriever = query_pinecone()
retriever_tool = create_retriever_tool(
    PM_retriever,
    "Product_manager",
    "Search information about product managment. ANSWER ALL PRODUCT MANAGMENT QUESTIONS WITH THIS TOOL!",

)
web_search = TavilySearchResults()
tools = [retriever_tool,web_search]

prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_functions_agent(llm,tools,prompt)
agent_excecutor =AgentExecutor(agent=agent,tools=tools,verbose = True)

class Input(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )
class Output(BaseModel):
    output: str



app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="Hepha API"

)

add_routes(
    app,
    agent_excecutor.with_types(input_type=Input,output_type=Output),
    path="/hepha"
)


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)