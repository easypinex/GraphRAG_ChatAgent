import logging
import warnings
from typing import Dict, List
warnings.simplefilter("ignore", DeprecationWarning)

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector

from ..llm_package.llm_module import EMB_MODEL
from .neo4j_config import *
from .cypher import *
from ..dto_package.chunk import Chunk

GRAPH = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)
LOGGER = logging.getLogger("TNPS")


def clear_all_nodes():
    """
    清除所有節點與index
    """
    GRAPH.query(DELETE_ALL_NODES)
    GRAPH.query(DELETE_ALL_VECTOR_INDEX1)
    GRAPH.query(DELETE_ALL_VECTOR_INDEX2)

def create_Node_Product(unique_filenames: List[str]):
    """
    在Neo4j圖數據庫中創建產品節點
    
    參數:
        unique_filenames: 不重複的產品名稱(文件名)
    """
    for filename in unique_filenames:
        params = {'product': filename}
        GRAPH.query(PRODUCT_QUERY, params=params)

def create_Node_Topic(topic_summary_dict: Dict[str, str]):
    """
    在Neo4j圖數據庫中創建主題節點
    
    參數:
        topic_summary_dict: 主題列表
        {
            "1": "商品條款的重點包括手術、疾病檢查、精神健康、醫療費用、住院及出院期間、投保項目、實際費用限額、診斷與治療、特定疾病、門診及住院醫療等。",
            ...
        }
    """
    for topic_num, topic_summary in topic_summary_dict.items():
        params = {'topic': topic_num, 'description': topic_summary}
        GRAPH.query(TOPIC_QUERY, params=params)

def create_Node_Chunk(all_file_chunks: List[Chunk]):
    """
    在Neo4j圖數據庫中創建內容塊節點
    
    參數:
        all_file_chunks: 包含所有內容塊的列表
    """
    for chunk in all_file_chunks:
        params = {
            'content': chunk.content,
            'filename': chunk.filename,
            'seg_list': chunk.segment_list,
            'topic_list': chunk.topic_list,
            'summary': chunk.summary,
        }
        GRAPH.query(CHUNK_QUERY, params=params)

def create_Relation_Topic_Chunks():
    """
    在Neo4j圖數據庫中創建主題與內容塊的關係
    """
    GRAPH.query(RELATION_QUERY_TC)

def create_Relation_Product_Chunks():
    """
    在Neo4j圖數據庫中創建產品與內容塊的關係
    """
    GRAPH.query(RELATION_QUERY_PC)

def create_VecotrIndex_content():
    """
    在Neo4j圖數據庫中創建內容塊的向量索引
    """
    Neo4jVector.from_existing_graph(
        embedding=EMB_MODEL,  # 嵌入模型
        url=NEO4J_URI,  # Neo4j數據庫URI
        username=NEO4J_USERNAME,  # 用戶名
        password=NEO4J_PASSWORD,  # 密碼
        database=NEO4J_DATABASE,  # 數據庫名稱
        index_name="emb_index",  # 索引名稱
        node_label="Chunk",  # 節點標籤
        embedding_node_property="contentEmbedding",  # 嵌入節點屬性
        text_node_properties=["content"],  # 文本節點屬性
        )
    GRAPH.refresh_schema()
    GRAPH.query(BUILD_VECTOR_INDEX_CONTENT)
    # GRAPH.query(SHOWINDEX)

def create_Node_RuleTopics(topic_summary_dict: Dict[str, str]):
    """
    在Neo4j圖數據庫中創建投保規則主題節點
    """
    for topic_num, topic_summary in topic_summary_dict.items():
        params = {'topic': topic_num, 'description': topic_summary}
        GRAPH.query(RULETOPIC_QUERY, params=params)

def create_Node_PageTable(all_file_chunks: List[Chunk]):
    """
    在Neo4j圖數據庫中創建頁面表格節點
    """
    for chunk in all_file_chunks:
        params = {
            'content': chunk.content,
            'filename': chunk.filename,
            'seg_list': chunk.segment_list,
            'topic_list': chunk.topic_list,
            'summary': chunk.summary,
            'page': chunk.page,
        }
        GRAPH.query(PAGETABLE_QUERY, params=params)

def create_Relation_RuleTpoic_Pagetable():
    """
    在Neo4j圖數據庫中創建投保規則主題與頁面表格的關係
    """
    GRAPH.query(RELATION_QUERY_RTPT)

def create_Relation_Product_Pagetable():
    """
    在Neo4j圖數據庫中創建產品與頁面表格的關係
    """
    GRAPH.query(RELATION_QUERY_PPT)

def create_VecotrIndex_pagetable():
    """
    在Neo4j圖數據庫中創建頁面表格的向量索引
    """
    Neo4jVector.from_existing_graph(
        embedding=EMB_MODEL,  # 嵌入模型
        url=NEO4J_URI,  # Neo4j數據庫URI
        username=NEO4J_USERNAME,  # 用戶名
        password=NEO4J_PASSWORD,  # 密碼
        database=NEO4J_DATABASE,  # 數據庫名稱
        index_name="emb_index_rule",  # 索引名稱
        node_label='PageTable',  # 節點標籤
        embedding_node_property='summaryEmbedding',  # 嵌入屬性名稱
        text_node_properties=['summary'])  # 文本屬性名稱
    GRAPH.refresh_schema()
    GRAPH.query(BUILD_VECTOR_INDEX_PAGETABLE)
    # GRAPH.query(SHOWINDEX)
    