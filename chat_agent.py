import os

from langchain_community.graphs import Neo4jGraph
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
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
from pydantic import BaseModel

from neo4j_module.twlf_neo4j_vector import TwlfNeo4jVector
from prompts.prompts import QUESTION_HISTORY_PROMPT, QUESTION_PROMPT

database = os.environ.get('NEO4J_DATABASE')
graph = Neo4jGraph(database=database)
embedding = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment='text-embedding-3-small',
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"]
)

lc_retrieval_query = """
WITH collect(node) as nodes
// Entity - Text Unit Mapping
WITH
collect {
    UNWIND nodes as n
    MATCH (n)<-[:HAS_ENTITY]->(c:__Chunk__)<-[:HAS_CHILD]->(p:__Parent__)
    WITH c, p, count(distinct n) as freq
    RETURN {content: p.content, source: p.source} AS chunkText
    ORDER BY freq DESC
    LIMIT $topChunks
} AS text_mapping,
// Entity - Report Mapping
collect {
    UNWIND nodes as n
    MATCH (n)-[:IN_COMMUNITY*]->(c:__Community__)
    WHERE c.summary is not null
    WITH c, c.rank as rank, c.weight AS weight
    RETURN c.summary 
    ORDER BY rank DESC, weight DESC
    LIMIT $topCommunities
} AS report_mapping,
// Outside Relationships 
collect {
    UNWIND nodes as n
    MATCH (n)-[r]-(m) 
    WHERE NOT m IN nodes and r.description is not null
    RETURN {description: r.description, sources: r.sources} AS descriptionText
    ORDER BY r.rank DESC, r.weight DESC 
    LIMIT $topOutsideRels
} as outsideRels,
// Inside Relationships 
collect {
    UNWIND nodes as n
    MATCH (n)-[r]-(m) 
    WHERE m IN nodes and r.description is not null
    RETURN {description: r.description, sources: r.sources} AS descriptionText
    ORDER BY r.rank DESC, r.weight DESC 
    LIMIT $topInsideRels
} as insideRels,
// Entities description
collect {
    UNWIND nodes as n
    match (n)
    WHERE n.description is not null
    RETURN {description: n.description, sources: n.sources} AS descriptionText
} as entities
// We don't have covariates or claims here
RETURN {Chunks: text_mapping, Reports: report_mapping, 
       Relationships: outsideRels + insideRels, 
       Entities: entities} AS text, 1.0 AS score, {} AS metadata
"""

vectorstore = Neo4jVector.from_existing_graph(embedding=embedding, 
                                    index_name="embedding",
                                    node_label='__Entity__', 
                                    embedding_node_property='embedding', 
                                    text_node_properties=['id', 'description'],
                                    retrieval_query=lc_retrieval_query)
topChunks = 3
topCommunities = 3
topOutsideRels = 10
topInsideRels = 10
topEntities = 10
local_search_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.9,
                   'k': topEntities,
                   'params': {
                        "topChunks": topChunks,
                        "topCommunities": topCommunities,
                        "topOutsideRels": topOutsideRels,
                        "topInsideRels": topInsideRels,
                    }},
    tags=['GraphRAG']
)
vectorstore = TwlfNeo4jVector.from_existing_graph(
                                    embedding=embedding, 
                                    index_name="chunk_index",
                                    node_label='__Chunk__', 
                                    embedding_node_property='embedding', 
                                    text_node_properties=['content'])
vector_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.9},
    tags=['RAG']
)

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
prompt = QUESTION_PROMPT

rag_chain = (
    {"context": vector_retriever, "question": RunnablePassthrough(), "graph_result": local_search_retriever}
    | prompt
    | llm
    | StrOutputParser()
)

# 定義上下文解析的Chain
contextualize_q_prompt = QUESTION_HISTORY_PROMPT

contextualize_chain = (
    contextualize_q_prompt
    | llm
    | StrOutputParser().with_config({
        'tags': ['contextualize_question']
    })
)

store = {}

    
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 使用 RunnableParallel 來組織多個並行查詢
context_and_search_chain = RunnableParallel(
    {
        "context": RunnableLambda(lambda inputs: vector_retriever.invoke(inputs)),
        "graph_result": RunnableLambda(lambda inputs: local_search_retriever.invoke(inputs)),
        "question": lambda x: x,  # 保留原始輸入
    }
)

rag_chain = (
    contextualize_chain
    | context_and_search_chain
    | prompt
    | llm
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



class ChatHistory(BaseModel):
    """Chat history with the bot."""
    question: str
    
conversational_rag_chain = (
  conversational_rag_chain | StrOutputParser()
).with_types(input_type=ChatHistory)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)