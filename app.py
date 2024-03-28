from fastapi import FastAPI
import uvicorn
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LCPC
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.pydantic_v1 import BaseModel
from langchain import hub



load_dotenv()

model = ChatOpenAI(model='gpt-3.5-turbo',temperature=0)



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
print(prompt.messages)
agent = create_openai_functions_agent(model,tools,prompt)

agent_excecutor =AgentExecutor(agent=agent,tools=tools,verbose = True)




app = FastAPI()


@app.post("/Hepha_api")
async def modify_string(input_string: str):
    print(type(input_string))
    print(input_string)
    result = agent_excecutor.invoke({'input':input_string})
    return result['output']

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)