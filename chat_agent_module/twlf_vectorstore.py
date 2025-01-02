import sys
import os

from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import AzureOpenAIEmbeddings

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    
from chat_agent_module.file_filter_retriever import FileFilterRetriever
from chat_agent_module.search_params import LocalSearchParamsDict
from neo4j_module.twlf_neo4j_vector import TwlfNeo4jVector

from chat_agent_module.file_metadata_retriever import FileMetadataRetriever

def get_localsearch_vectorstore(embedding) -> TwlfNeo4jVector:
    lc_retrieval_query = """
// "CALL db.index.vector.queryNodes($index, $k * $ef, $embedding) "
// "YIELD node, score "
// "WITH node, score LIMIT $k "
// 1) 先 collect 好所有的 nodes 並收集對應的 fileIds
WITH collect(node) AS nodes
MATCH (node)<-[:HAS_ENTITY]-(c:__Chunk__)<-[:HAS_CHILD]-(p:__Parent__)-[:PART_OF]-(d:__Document__)
WHERE $fileIds IS NULL OR d.file_task_id IN $fileIds
WITH nodes, collect(DISTINCT d.file_task_id) AS docIds

// 2) 以下才進入原本的子查詢區塊。
//   注意這裡每個 collect { ... } 都要在同一個 WITH 流程下，
//   或者使用多段 WITH 依序聚合都可以，關鍵是最後要把 docIds 留到最後。
WITH
    // Chunks
    collect {
        UNWIND nodes AS n
        MATCH (n)<-[:HAS_ENTITY]-(c:__Chunk__)<-[:HAS_CHILD]-(p:__Parent__)-[:PART_OF]-(d:__Document__)
        WHERE $fileIds IS NULL OR d.file_task_id IN $fileIds
        WITH c, p, count(distinct n) as freq
        RETURN  p.content AS chunkText
        ORDER BY freq DESC
        LIMIT $topChunks
    } AS text_mapping,
    // Entity - Report Mapping
    collect {
        UNWIND nodes AS n
        MATCH (n)-[:IN_COMMUNITY*]->(com:__Community__)
        WHERE com.summary IS NOT NULL
        WITH com, com.rank AS rank, com.weight AS weight
        RETURN com.summary
        ORDER BY rank, weight DESC
        LIMIT $topCommunities
    } AS report_mapping,
    // Outside Relationships
    collect {
        UNWIND nodes AS n
        MATCH (n)-[r]-(m)
        WHERE NOT m IN nodes AND r.description IS NOT NULL
        RETURN r.description AS descriptionText
        ORDER BY r.rank DESC, r.weight DESC
        LIMIT $topOutsideRels
    } AS outsideRels,
    // Inside Relationships
    collect {
        UNWIND nodes AS n
        MATCH (n)-[r]-(m)
        WHERE m IN nodes AND r.description IS NOT NULL
        RETURN r.description AS descriptionText
        ORDER BY r.rank DESC, r.weight DESC
        LIMIT $topInsideRels
    } AS insideRels,
    // Entities description
    collect {
        UNWIND nodes AS n
        MATCH (n)
        WHERE n.description IS NOT NULL
        RETURN n.description AS descriptionText
    } AS entities,
    docIds   // 這裡把前面收集的 docIds 一起帶下來
RETURN
{
    // 這裡的 text 是輸出的主要內容
    Chunks: text_mapping,
    Reports: report_mapping,
    Relationships: outsideRels + insideRels,
    Entities: entities
} AS text,
1.0 AS score,
// 這裡就是我們要在 metadata 裡面帶出所有出現的 fileId
{ fileIds: docIds } AS metadata
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
    local_search_retriever = FileMetadataRetriever(local_search_retriever)
    return local_search_retriever

def get_baseline_vectorstore(embedding) -> TwlfNeo4jVector:
    vectorstore: Neo4jVector = TwlfNeo4jVector.from_existing_graph(
                                        embedding=embedding, 
                                        index_name="chunk_index",
                                        node_label='__Chunk__', 
                                        embedding_node_property='embedding', 
                                        text_node_properties=['content'])
    return vectorstore

def get_baseline_retriever(embedding, score_threshold: float = 0.9) -> TwlfNeo4jVector:
    vectorstore: Neo4jVector = get_baseline_vectorstore(embedding)
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': score_threshold},
        tags=['BaselineRAG']
    )
    baseline_retriever = FileFilterRetriever(vector_retriever, node_label='__Chunk__')
    baseline_retriever = FileMetadataRetriever(baseline_retriever)
    return baseline_retriever

if __name__ == "__main__":
    grpah = Neo4jGraph()
    embedding = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment='text-embedding-3-small',
        openai_api_version='2023-05-15'
    )
    # localsearch_retriever = get_localsearch_retriever(embedding)
    # print(localsearch_retriever.invoke({
    #                                 "question": "台灣人壽",
    #                                 "inputs": {
    #                                     # "fileIds": [1],
    #                                     # "fileIds": None
    #                                 }
    #                             }))
    
    
    baseline_retriever: TwlfNeo4jVector = get_baseline_retriever(embedding, 0.9)
    print(baseline_retriever.invoke({
        "question": """主契約效力停止時，要保人不得單獨申請恢復本附約之效力。
基於保戶服務，本公司於保險契約停止效力後至得申請復效之期限屆滿前三個月，將以書面、電子郵件、簡訊或其
他約定方式擇一通知要保人有行使申請復效之權利，並載明要保人未於約定期限屆滿前恢復保單效力者，契約效力
將自約定期限屆滿之日翌日上午零時起終""",
        "inputs": {
            # "fileIds": [1],
        }
    }))