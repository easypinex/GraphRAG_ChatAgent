import os
from typing import List, Optional

from langchain_openai import AzureOpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers.string import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema.runnable import Runnable

from chat_agent_module.file_metadata_search import FileMetadataSearch
from chat_agent_module.file_scope_selector import FileScopeSelector
from chat_agent_module.remove_metadata import RemoveMetadata
from chat_agent_module.twlf_vectorstore import get_baseline_retriever, get_localsearch_retriever
from prompts.prompts import QUESTION_HISTORY_PROMPT, QUESTION_PROMPT

database = os.environ.get('NEO4J_DATABASE')
graph = Neo4jGraph(database=database)
embedding = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment='text-embedding-3-small',
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"]
)
local_search_retriever = get_localsearch_retriever(embedding)
local_search_retriever_chain: Runnable = local_search_retriever | FileMetadataSearch()
vector_retriever = get_baseline_retriever(embedding, score_threshold=0.85)
vector_retriever_chain: Runnable = vector_retriever | FileMetadataSearch()
    
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME_MAIN"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0
)

large_llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME_LARGE"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0
)

# rag_chain = (
#     {"context": vector_retriever, "question": RunnablePassthrough(), "graph_result": local_search_retriever}
#     | prompt
#     | llm
# )

# 定義上下文解析的Chain
contextualize_chain = (
    QUESTION_HISTORY_PROMPT
    | large_llm
    | StrOutputParser().with_config({
        'tags': ['contextualize_question'],
        'name': 'FileMetadataSearch'
    })
)

store = {}

remover = RemoveMetadata()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 使用 RunnableParallel 來組織多個並行查詢
context_and_search_chain = RunnableParallel(
    {
        "context": RunnableLambda(
            lambda inputs: 
                remover.invoke(vector_retriever_chain.invoke(inputs))
        ),
        "graph_result": RunnableLambda(
            lambda inputs: 
                remover.invoke(local_search_retriever_chain.invoke(inputs))
        ),
        "question": lambda inputs: inputs.get("question"),  # 保留原始問題
    }
)

rag_chain = (
    {"question": contextualize_chain, "inputs": RunnablePassthrough()}
    | FileScopeSelector(large_llm)
    | context_and_search_chain
    | QUESTION_PROMPT
    | large_llm
    | StrOutputParser().with_config({
        "tags": ['final_output']
    })
)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)



class ChatInput(BaseModel):
    """Chat history with the bot."""
    question: str
    fileIds: List[int] = Field(default=[], description="Optional list of file IDs")
    
    
conversational_rag_chain = (
  conversational_rag_chain | StrOutputParser()
).with_types(input_type=ChatInput)

# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route

add_routes(
    app,
    conversational_rag_chain
)

# 允許cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],          # 允許的來源列表
    allow_methods=["*"],            # 允許的 HTTP 方法
    allow_headers=["*"],            # 允許的 HTTP 標頭
)

ssl_keyfile = os.getenv("CHAT_AGENT_SSL_PRIVATE_KEY")
ssl_certfile = os.getenv("CHAT_AGENT_SSL_PUBLIC_KEY")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", 
                port=8000,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile
                )