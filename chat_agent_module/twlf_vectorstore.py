import sys
import os

from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import AzureOpenAIEmbeddings

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    
from chat_agent_module.file_filter_retriever import FileFilterRetriever
from chat_agent_module.search_params import LocalSearchParamsDict
from neo4j_module.twlf_neo4j_vector import TwlfNeo4jVector

def get_localsearch_vectorstore(embedding) -> TwlfNeo4jVector:
    lc_retrieval_query = """
        // MATCH (n:__Entity__) WITH collect(n) AS nodes return nodes
        WITH collect(node) as nodes
        WITH
        collect {
            UNWIND nodes as n
            MATCH (n)<-[:HAS_ENTITY]->(c:__Chunk__)<-[:HAS_CHILD]->(p:__Parent__)-[:PART_OF]->(d:__Document__)
            WHERE $fileIds IS NULL OR d.file_task_id IN $fileIds
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
            ORDER BY rank, weight DESC
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

    vectorstore = TwlfNeo4jVector.from_existing_graph(embedding=embedding, 
                                        index_name="embedding",
                                        node_label='__Entity__', 
                                        embedding_node_property='embedding', 
                                        text_node_properties=['id', 'description'],
                                        retrieval_query=lc_retrieval_query)
    return vectorstore

def get_localsearch_retriever(embedding) -> TwlfNeo4jVector:
    vectorstore: Neo4jVector = get_localsearch_vectorstore(embedding)
    topEntities = 10
    search_params = LocalSearchParamsDict.default()
    local_search_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.9,
                    'k': topEntities,
                    'params': search_params
                    },
        tags=['GraphRAG']
    )
    local_search_retriever = FileFilterRetriever(local_search_retriever)
    return local_search_retriever

def get_baseline_vectorstore(embedding) -> TwlfNeo4jVector:
    vectorstore: Neo4jVector = TwlfNeo4jVector.from_existing_graph(
                                        embedding=embedding, 
                                        index_name="chunk_index",
                                        node_label='__Chunk__', 
                                        embedding_node_property='embedding', 
                                        text_node_properties=['content'])
    return vectorstore

def get_baseline_retriever(embedding) -> TwlfNeo4jVector:
    vectorstore: Neo4jVector = get_baseline_vectorstore(embedding)
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.9},
        tags=['RAG']
    )
    baseline_retriever = FileFilterRetriever(vector_retriever)
    return baseline_retriever

if __name__ == "__main__":
    grpah = Neo4jGraph()
    embedding = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment='text-embedding-3-small',
        openai_api_version='2023-05-15'
    )
    vectorstore = get_localsearch_vectorstore(embedding)
    search_params = LocalSearchParamsDict.default()
    search_params['fileIds'] = ['1'] # 這裡特別設置不存在的檔案名稱, 後續驗證更新
    topEntities = 3

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
                        'score_threshold': 0.9,
                        'k': topEntities,
                        'params': search_params
                    },
        tags=['GraphRAG']
    )
    dynamic_retriver = FileFilterRetriever(retriever)
    print(dynamic_retriver.invoke({
                                    "question": "台灣人壽",
                                    "inputs": {
                                        "fileIds": [1],
                                        # "fileIds": None
                                    }
                                }))
    # vectorstore: Neo4jVector = TwlfNeo4jVector.from_existing_graph(
    #                                 embedding=embedding, 
    #                                 index_name="chunk_index",
    #                                 node_label='__Chunk__', 
    #                                 embedding_node_property='embedding', 
    #                                 text_node_properties=['content'])
#     vector_retriever = vectorstore.as_retriever(
#         search_type="similarity_score_threshold",
#         search_kwargs={'score_threshold': 0.9},
#         tags=['RAG']
#     )
#     vector_retriever = FileFilterRetriever(vector_retriever)
#     print(vector_retriever.invoke({
#         "question": """乳房手術項目給付比例如下：

# 1. 單純乳房切除術(單側) 給付比例為 12%
# 2. 單純乳房切除術(雙側) 給付比例為 17%
# 3. 乳癌根治切除術(單側) 給付比例為 27%
# 4. 乳癌根治切除術(雙側) 給付比例為 39%
# 附表三：放射線治療項目及費用表

# 放射線治療項目給付比例如下：

# 1. 照射治療規劃及劑量（每次）：7%
# 2. 初步或定位照相（每張）：2%
# 3. 鈷六十照射（每次）：3%
# 4. 直線加速器照射治療（每次）：5%""",
#         "inputs": {
#             "fileIds": ["bbe9f0d5-a3a3-49c2-92e7-07367d14ebd1"],
#         }
#     }))