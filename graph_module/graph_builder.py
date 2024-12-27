"""整理建構圖樹方法"""
import os
from typing import Any, Dict, List
from uuid import uuid4 as uuid

from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import (Document, GraphDocument,
                                                       Node, Relationship)
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents.transformers import BaseDocumentTransformer
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from .dto.simple_graph import DocChunk, GraphGraphDetails, SimpleGraph
from .twlf_llm_graph_transformer import TwlfLlmGraphTransformer

from tqdm import tqdm


class GraphBuilder:
    def __init__(self, graph: Neo4jGraph, max_thread=5):
        self.graph = graph
        self._tag_node_id_map = {}  # tag, node_id, 用以記憶每個tag node 的 id, 減少查詢
        self._max_thread = max_thread
        self._source_doc_map = {}

    def build_chunk_graph_with_parent_child(self, docs_pages: List[List[Document]], parent_split_kwarg=None, child_split_kwarg=None, document_additional_properties=None) -> SimpleGraph:
        """
        將每份文件的內容處理產生 SimpleGraph
        """
        if len(docs_pages) == 0:
            return
        
        if parent_split_kwarg is None:
            parent_split_kwarg = {'chunk_size': 1200, 'chunk_overlap': 200, 'separators': ['\n\n', '【',  '。']}
        if child_split_kwarg is None:
            child_split_kwarg = {'chunk_size': 300, 'chunk_overlap': 30, 'separators': ['\n\n','【', '。', ]}
        if document_additional_properties is None:
            document_additional_properties = {}

        chunks = [] # final result
        result: SimpleGraph = SimpleGraph()
        for doc_pages in docs_pages:
            doc = self._merge_all_pages(doc_pages)
            doc.page_content = self._bad_chars_clear(doc.page_content)

            # 建立/取得 根 Document / Node
            root_source = doc.metadata.get('source') # 檔名
            root_document_dict = self._get_source_document(root_source, total_page=len(doc_pages))

            root_graph_document: GraphDocument = root_document_dict['graph_document']
            root_document: Document = root_document_dict['document']
            root_document.metadata.update(document_additional_properties)
            root_node: Node = root_document_dict['node']

            # 建立分割器
            parent_split = RecursiveCharacterTextSplitter(**parent_split_kwarg)
            child_split = RecursiveCharacterTextSplitter(**child_split_kwarg)

            # 建立 父 Document
            parent_docs = parent_split.split_documents([doc])
            for parent_doc in parent_docs:
                # 建立父 Node
                properties = {
                    'source': root_source,
                    'content': parent_doc.page_content
                }
                parent_node = Node(id=str(uuid()), type='__Parent__', properties=properties)
                
                # 關聯 主文件 -> 父節點 
                root_graph_document.nodes.append(parent_node)
                
                # 建立 關係 root_node -> parent_node
                root_graph_document.relationships.append(Relationship(source=parent_node, target=root_node, type='PART_OF'))
                # 開始建立子節點 (Chunk)
                child_docs = child_split.split_documents([parent_doc])
                for child_doc in child_docs:
                    # 建立子 Node
                    properties = {
                        'source': root_source,
                        'content': child_doc.page_content
                    }
                    child_node = Node(id=str(uuid()), type='__Chunk__', properties=properties)
                    
                    # 關聯 父節點 -> 子節點 
                    root_graph_document.nodes.append(child_node)
                    
                    # 建立 關係 parent_node -> child_node
                    root_graph_document.relationships.append(Relationship(source=parent_node, target=child_node, type='HAS_CHILD'))
                    chunks.append(DocChunk(chunk_id=child_node.id, chunk_doc=child_doc))

            result.details.append(GraphGraphDetails(root_document=root_document, root_node=root_node, root_graph_document=root_graph_document))
        result.chunks = chunks
        return result
        
    def save_simple_graph_to_neo4j(self, details: List[GraphGraphDetails] = None):
        """
        以 GraphDocument 的格式將圖樹存入 Neo4j
        document node 會因為 relation 當中的 source node 的關係而自動被建立, 但是 document node 本身的 property 需要自行更新
        """
        for detail in details:
            root_graph_document, root_document, root_node = detail.root_graph_document, detail.root_document, detail.root_node
            self.graph.add_graph_documents([root_graph_document])
            self.update_node_properties(root_node.id, root_document.metadata)

    def _get_source_document(self, source: str, total_page=None):
        """
        從 source(檔案路徑) 取得 GraphGraphDetails 所需的 cached 資料 (dict 格式), 如果沒有就創建一個空的

        Args:
            source (str): 檔案名稱 , 或任何 String

        Returns:
            GraphGraphDetails (dict): 
                {
                    'node': Node(id: str, type='__Document__'), 
                    'document': Document(page_content="", metadata={id, filename, file_path, total_page_num}), 
                    'graph_document': GraphDocument(nodes=[], relationships=[], source=Document), 
                }
        """        
        if source in self._source_doc_map:
            self._source_doc_map[source]['document'].metadata['total_page_num'] += 1
            return self._source_doc_map[source]
        
        document_dict = {}  # keys [document, node]
        document_node = Node(id=str(uuid()), type='__Document__')
        document_dict['node'] = document_node

        filename = os.path.basename(source)
        doc_properties = {
            'id': document_node.id,
            'filename': filename,
            'file_path': source,
            'total_page_num': 1 if total_page is None else total_page,
        }
        document_dict['document'] = Document(page_content="", metadata=doc_properties)
        
        graph_document = GraphDocument(
            nodes=[], relationships=[], source=document_dict['document'])
        document_dict['graph_document'] = graph_document
        self._source_doc_map[source] = document_dict

        return document_dict

    def _merge_all_pages(self, docs: List[Document]) -> Document:
        """
        將多個 Document(page_coutent=..., metadata={'source': '檔名', 'page_number': int})
        整併成一個 Document(page_coutent=..., metadata={'source': '檔名', 'total_page_num': int})
        """
        if len(docs) == 0:
            return None
        doc = Document(page_content="\n".join([doc.page_content for doc in docs]), metadata={'source': docs[0].metadata.get('source'), 'total_page_num': len(docs)})
        return doc
    
    def update_node_properties(self, node_id: str, node_properties: dict) -> List[Dict[str, Any]] | None:
        """
        以 node_id 將該 node 的 property 做 更新(如果存在property)/新增(如果不存在property)
        """
        if len(node_properties) == 0:
            return None
        
        set_query = ''
        for key in node_properties.keys():
            set_query += f'n.{key} = ${key}, '
        set_query = set_query[:-2] # 去掉最後的 ', '

        temp = f'''
                MATCH (n) WHERE n.id = '{node_id}'
                SET {set_query}
                RETURN n
                '''
        
        return self.graph.query(temp, node_properties)

    def _bad_chars_clear(self, text="", bad_chars: List[str] | None = None):
        """
        清理文本中的不良字符[", ', ...]
        """
        if bad_chars is None:
            bad_chars = ['"', "'", "..."]
        for bad_char in bad_chars:
            if bad_char == '\n':
                text = text.replace(bad_char, ' ')
            else:
                text = text.replace(bad_char, '')
        return text.strip()

    def get_chunk_and_graph_document(self, graph_document_list: List[GraphDocument]) -> List[dict]:
        '''
        格式整理
        原本 GraphDocument 以 Entity 為單位, 紀錄這些 Entity 是從哪些 chunk_ids 來的
        整理成以 chunk_id 為單位, 紀錄這些 chunk_id 有哪些 Entity

        params:
            graph_document_list: 每個 Chunk 擷取出來的 Entity [GraphDocument{}, ...]
        return:
            lst_chunk_chunkId_document: [{'graph_doc': GraphDocument, 'chunk_id': str}, ...]
        '''
        logging.info(
            "creating list of chunks and graph documents in get_chunk_and_graphDocument func")
        lst_chunk_chunkId_document = []
        for graph_document in graph_document_list:
            for chunk_id in graph_document.source.metadata['combined_chunk_ids']:
                lst_chunk_chunkId_document.append(
                    {'graph_doc': graph_document, 'chunk_id': chunk_id})

        return lst_chunk_chunkId_document

    def merge_relationship_between_chunk_and_entites(self, graph_documents_chunk_chunk_Id: list) -> List[dict]:
        '''
        將 chunk 與 節點 的關係以 cypher 語法建立 relationship

        params:
            graph_documents_chunk_chunk_Id: [{'graph_doc': GraphDocument, 'chunk_id': str}, ...]
        '''
        logging.info(
            "Create HAS_ENTITY relationship between chunks and entities")
        
        batch_data = []
        for graph_doc_chunk_id in graph_documents_chunk_chunk_Id:
            for node in graph_doc_chunk_id['graph_doc'].nodes:
                query_data = {
                    'chunk_id': graph_doc_chunk_id['chunk_id'],
                    'node_type': node.type,
                    'node_id': node.id,
                    'source': graph_doc_chunk_id['graph_doc'].source.metadata['source']
                }
                batch_data.append(query_data)
                # node_id = node.id
                # Below query is also unable to change as parametrize because we can't make parameter of Label or node type
                # https://neo4j.com/docs/cypher-manual/current/syntax/parameters/
                # graph.query('MATCH(c:Chunk {'+chunk_node_id_set.format(graph_doc_chunk_id['chunk_id'])+'}) MERGE (n:'+ node.type +'{ id: "'+node_id+'"}) MERGE (c)-[:HAS_ENTITY]->(n)')

        if batch_data:
            unwind_query = """
                        UNWIND $batch_data AS data
                        MATCH (c:__Chunk__ {id: data.chunk_id})
                        CALL apoc.merge.node([data.node_type], {id: data.node_id}) YIELD node AS n
                        MERGE (c)-[:HAS_ENTITY]->(n)
                        SET n.sources = apoc.coll.union(coalesce(n.sources, []), [data.source])
                    """
            self.graph.query(unwind_query, params={"batch_data": batch_data})

    def get_entities_graph_from_llm(self, llm, chunk_id_with_chunk_doc_list: list[DocChunk], allowedNodes: list[str], allowedRelationship: list[str]) -> List[GraphDocument]:
        '''
        提供所有 Chunk 由 LLM 提取 Entity, 以及 Entity 之間的 Relationship

        params:
            llm: LLM 模型
            chunkId_chunkDoc_list: [DocChunk{'chunk_id': str, 'chunk_doc': Document}, ...]
            allowedNodes: List[str] 允許的節點類型(Label)
            allowedRelationship: List[str] 允許的關係

        return:
            graph_document_list: List[GraphDocument]
        '''
        # 將 DocChunk 轉換成 Document
        combined_chunk_document_list = self._get_combined_chunks(chunk_id_with_chunk_doc_list)

        # 輸入 [Document] 取得 [GraphDocument] 代表每個 Chunk 提取出來的 Entity
        entity_graph_document_list = self._get_graph_document_list(llm, combined_chunk_document_list, allowedNodes, allowedRelationship, max_retry=0)

        # 清理 不存在/不合理 的 Relationship (根據 source, target 是否存在於 nodes)
        self._clean_not_exists_rel(entity_graph_document_list)

        # 新增 uuid 以利未來建立 Commnuity 時, 識別是否是同一群 Node(Entity) 與 Relationship
        self._mark_node_rel_with_uuid(entity_graph_document_list)

        return entity_graph_document_list

    def _get_combined_chunks(self, chunkId_chunkDoc_list: list[DocChunk], chunks_to_combine=1) -> List[Document]:
        """
        主要目的是把 list[DocChunk] 轉換成 list[Document], 還可以使用參數將多個 DocChunk 合併成一個 Document

        Args:
            chunkId_chunkDoc_list (List[Dict{'chunk_id': ..., 'chunk_doc': Document}, ...]): Chunk列表
            chunks_to_combine (int, optional): 幾個 Chunk 進行合併. 不合併 Defaults to 1.

        Returns:
            List[Document]: 合併後的 Document (metadtata{'source': ..., 'total_page_num': ..., 'combined_chunk_ids': [...]})
        """        
        logging.info(
            f"Combining {chunks_to_combine} chunks before sending request to LLM")
        
        combined_chunk_document_list = []

        combined_chunks_page_content = [
            "".join(
                document.chunk_doc.page_content
                for document in chunkId_chunkDoc_list[i: i + chunks_to_combine]
            )
            for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
        ]
        combined_chunks_ids = [
            [
                document.chunk_id
                for document in chunkId_chunkDoc_list[i: i + chunks_to_combine]
            ]
            for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
        ]
        combined_metadatas = [
            chunkId_chunkDoc_list[i].chunk_doc.metadata for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
        ]

        for i in range(len(combined_chunks_page_content)):
            metadata = combined_metadatas[i]
            metadata['combined_chunk_ids'] = combined_chunks_ids[i]
            combined_chunk_document_list.append(
                Document(
                    page_content=combined_chunks_page_content[i],
                    metadata=metadata,
                )
            )
        return combined_chunk_document_list

    def _get_graph_document_list(
        self, llm, combined_chunk_document_list: List[Document], allowedNodes, allowedRelationship, max_retry=0
    ) -> List[GraphDocument]:
        """
        由 LLM 取得 Entity, 以及 Entity 之間的 Relationship

        params:
            llm: LLM Model
            combined_chunk_document_list (List[Document]): 合併後的 Chunk Document (metadtata{'source': ..., 'total_page_num': ..., 'combined_chunk_ids': [...]})
            allowedNodes ([str]): 允許產生的 Entity (可多個)
            allowedRelationship ([str]): 允許的產生的 Relationship (可多個)
            max_retry (int, optional): retry 次數. Defaults to 0.

        return:
            List[GrapDocument]: node 是 Entity, relationship 是 Entity 之間的關係, source 是 chunk document
        """
        
        futures = []
        graph_document_list = []
        node_properties = ["description"]
        relationship_properties = ["description"]

        llm_transformer = TwlfLlmGraphTransformer(
            llm=llm,
            allowed_nodes=allowedNodes,
            allowed_relationships=allowedRelationship,
            node_properties=node_properties,                    # 允許 LLM 提取 node 的哪些 property 來產生 entity 與 relationship
            relationship_properties=relationship_properties     # 允許 LLM 提取 relationship 的哪些 property 來產生 entity 與 relationship
        )

        # 以 multi-threading 進行透過 LLM 從 Chunk 中取得 Entity 與 Relationship
        futures_to_chunk_doc: Dict[concurrent.futures.Future, Document] = {}
        failed_documents = []
        with ThreadPoolExecutor(max_workers=self._max_thread) as executor:
            for chunk in combined_chunk_document_list:
                chunk_doc = Document(
                    page_content=chunk.page_content.encode("utf-8"), metadata=chunk.metadata
                )
                future = executor.submit(
                    llm_transformer.convert_to_graph_documents, [chunk_doc]
                )
                futures.append(future)
                futures_to_chunk_doc[future] = chunk  # 關聯 future 和 chunk_doc    

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(combined_chunk_document_list)):
                try:
                    graph_documents:List[GraphDocument] = future.result(timeout=5 * 60) # 一個Chunk最多等待5分鐘
                    graph_doc = graph_documents[0] # 送進 LLM 是 [Document], 回傳是 [GraphDocument]
                    graph_document_list.append(graph_doc)
                except Exception as e:
                    chunk = futures_to_chunk_doc[future]
                    failed_documents.append(chunk)
                    print(f"Error processing document: {chunk}")
                    print(e)

        # 假如有失敗的 chunk 且 max_retry > 0, 則重新嘗試
        if len(failed_documents) > 0 and max_retry > 0:
            graph_document_list += self._get_graph_document_list(llm, failed_documents, allowedNodes, allowedRelationship, max_retry-1)
        
        return graph_document_list

    def _mark_node_rel_with_uuid(self, graph_document_list: list[GraphDocument]) -> list[GraphDocument]:
        '''
        為每一個 node 與 rel 的 properties 新增 uuid, uuid_hash
        uuid 是為了在未來建立 Community 時, 識別是否是與過去是同一群 Node(Entity) 與 Relationship, 如果一致則可用快取
        以及建立 Community 時順序的基準點 (uuid_hash)
        '''
        for graph_document in graph_document_list:
            for node in graph_document.nodes:
                uuid_obj = uuid()  # 生成一個隨機 UUID
                uuid_str = str(uuid_obj)  # 將 UUID 轉換為字串
                node.properties['uuid'] = uuid_str  # 存儲原始 UUID 字串
                node.properties['uuid_hash'] = uuid_obj.int % (2**31 - 1) # Neo4j 只能儲存 int, 數字太大Neo4j會報錯
            for relationship in graph_document.relationships:
                uuid_obj = uuid()  # 生成一個隨機 UUID
                uuid_str = str(uuid_obj)  # 將 UUID 轉換為字串
                relationship.properties['uuid'] = uuid_str
                relationship.properties['uuid_hash'] = uuid_obj.int % (2**31 - 1)
        return graph_document_list
    
    def _clean_not_exists_rel(self, graph_document_list: list[GraphDocument]):
        """
        只保留 source 與 target 都存在於 graph_document.nodes 中的 relationship
        """
        for graph_document in graph_document_list:
            exists_node = {node.id for node in graph_document.nodes}
            relationships = graph_document.relationships

            write_index = 0
            for read_index in range(len(relationships)):
                relationship = relationships[read_index]
                if relationship.source.id in exists_node and relationship.target.id in exists_node:
                    relationships[write_index] = relationships[read_index]  # 保留有效元素
                    write_index += 1

            # 刪除多餘元素
            del relationships[write_index:]
        