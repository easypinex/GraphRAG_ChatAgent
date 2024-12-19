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
        if len(docs_pages) == 0:
            return
        chunks = [] # final result
        if parent_split_kwarg is None:
            parent_split_kwarg = {'chunk_size': 1200, 'chunk_overlap': 200, 'separators': ['\n\n', '【',  '。']}
        if child_split_kwarg is None:
            child_split_kwarg = {'chunk_size': 300, 'chunk_overlap': 30, 'separators': ['\n\n','【', '。', ]}
        if document_additional_properties is None:
            document_additional_properties = {}
        result: SimpleGraph = SimpleGraph()
        for doc_pages in docs_pages:
            doc = self._merge_all_pages(doc_pages)
            doc.page_content = self._bad_chars_clear(doc.page_content)
            # 建立/取得 根 Document / Node
            root_source = doc.metadata.get('source')
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
                root_document_dict['pre_node'] = parent_node
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
            result.details.append(GraphGraphDetails(root_document, root_node, root_graph_document))
        result.chunks = chunks
        return result
        
    def save_simple_graph_to_neo4j(self, details: List[GraphGraphDetails] = None):
        for detail in details:
            root_graph_document, root_document, root_node = detail.root_graph_document, detail.root_document, detail.root_node
            self.graph.add_graph_documents([root_graph_document])
            self.update_node_properties(root_node.id, root_document.metadata)

    def _get_source_document(self, source: str, total_page=None):
        """
        從 source(檔案路徑) 取得 document
        Args:
            source (str): 檔案名稱 , 或任何 String

        Returns:
            document (dict): 
                {'document': Document, 
                'node': Node, 
                'page': int, 
                'graph_document': GraphDocument, 
                'pre_node': Document}
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
        document_dict['pre_node'] = document_dict['node']
        self._source_doc_map[source] = document_dict
        return document_dict

    def _merge_all_pages(self, docs: List[Document]) -> Document:
        if len(docs) == 0:
            return None
        doc = Document(page_content="\n".join([doc.page_content for doc in docs]), metadata={'source': docs[0].metadata.get('source'), 'total_page_num': len(docs)})
        return doc
    
    def update_node_properties(self, node_id: str, node_properties: dict) -> List[Dict[str, Any]] | None:
        if len(node_properties) == 0:
            return None
        set_query = ''
        for key in node_properties.keys():
            set_query += f'n.{key} = ${key}, '
        set_query = set_query[:-2]
        temp = f'''
                MATCH (n) WHERE n.id = '{node_id}'
                SET {set_query}
                RETURN n
                '''
        return self.graph.query(temp, node_properties)

    def _bad_chars_clear(self, text="", bad_chars: List[str] | None = None):
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
        將圖樹中的 chunk 與 節點 提取出來, 提供陣列

        params:
            graph_document_list: 圖陣列
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
        將 chunk 與 節點 的關係於資料庫中串在一起

        params:
            graph_documents_chunk_chunk_Id: [{'graph_doc': GraphDocument, 'chunk_id': str}, ...]
        '''
        batch_data = []
        logging.info(
            "Create HAS_ENTITY relationship between chunks and entities")
        chunk_node_id_set = 'id:"{}"'
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
        由 LLM 取得圖樹

        params:
            llm: LLM 模型
            chunkId_chunkDoc_list: [{'chunk_id': str, 'chunk_doc': Document}, ...]
            allowedNodes: List[str] 允許的節點類型(Label)
            allowedRelationship: List[str] 允許的關係

        return:
            graph_document_list: List[GraphDocument]
        '''
        combined_chunk_document_list = self._get_combined_chunks(
            chunk_id_with_chunk_doc_list)
        graph_document_list = self._get_graph_document_list(
            llm, combined_chunk_document_list, allowedNodes, allowedRelationship, max_retry=0
        )
        # 清理不存在/不合理的Relationship
        self._clean_not_exists_rel(graph_document_list)
        # 新增uuid以利未來建立 Commnuity 時, 識別是否是同一群 Node(Entity) 與 Relationship
        self._mark_node_rel_with_uuid(graph_document_list)
        return graph_document_list

    def _get_combined_chunks(self, chunkId_chunkDoc_list: list[DocChunk], chunks_to_combine=1) -> List[Document]:
        """_summary_

        Args:
            chunkId_chunkDoc_list (List[Dict[{'chunk_id': ..., 'chunk_doc': Document}]]): Chunk列表
            chunks_to_combine (int, optional): 幾個 Chunk 進行合併. 不合併 Defaults to 1.

        Returns:
            List[Document]: 合併後的 Document
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
        
        futures = []
        graph_document_list = []
        node_properties = ["description"]
        relationship_properties = ["description"]
        llm_transformer = TwlfLlmGraphTransformer(
            llm=llm,
            node_properties=node_properties,
            allowed_nodes=allowedNodes,
            allowed_relationships=allowedRelationship,
            relationship_properties=relationship_properties
        )
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
                    graph_doc = graph_documents[0]
                    graph_document_list.append(graph_doc)
                except Exception as e:
                    chunk = futures_to_chunk_doc[future]
                    failed_documents.append(chunk)
                    print(f"Error processing document: {chunk}")
                    print(e)
        if len(failed_documents) > 0 and max_retry > 0:
            graph_document_list += self._get_graph_document_list(llm, failed_documents, allowedNodes, allowedRelationship, max_retry-1)
        return graph_document_list

    def _mark_node_rel_with_uuid(self, graph_document_list: list[GraphDocument]) -> list[GraphDocument]:
        '''
        uuid是為了在未來建立 Community 時, 識別是否是與過去是同一群 Node(Entity) 與 Relationship, 如果一致則可用快取
        以及建立Community時順序的基準點 (uuid_hash)
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
        