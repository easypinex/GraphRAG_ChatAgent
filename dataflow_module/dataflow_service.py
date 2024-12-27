import sys
import os

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    
import json

from langchain_openai import AzureChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument
from langchain_openai import AzureOpenAIEmbeddings

from database import db_session
from models.file_task import FileTask
from file_module.file_service import FileService
from graph_module.graph_builder import GraphBuilder
from graph_module.knowledge_service import KnowledgeService
from graph_module.dto.simple_graph import SimpleGraph
from dataflow_module.rabbitmq_task import QueueTaskDict
from dataflow_module.rabbitmq_sender import publish_queue_message_sync
from graph_module.dto.summary_info_dict import SummaryInfoDict
from graph_module.dto.comnuity_info_dict import Neo4jCommunityInfoDict, compare_community_lists
from minio_module.minio_service import MinioService, minio_service
from neo4j_module.neo4j_object_serialization import dict_to_graph_document, graph_document_to_dict
from neo4j_module.neo4j_backup_restore import backup_neo4j_to_dict, restore_neo4j_from_dict
from graph_module.dto.duplicate_info_dict import DuplicateInfoDict


class DataflowService:
    def __init__(self, parse_file_using_llm: bool = True):
        self._llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0
        )
        self._embedding = AzureOpenAIEmbeddings(
            model="text-embedding-3-small",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment='text-embedding-3-small',
            openai_api_version='2023-05-15'
        )
        # 搭配llm可以將文件表格轉為自然語言
        self._file_service = FileService(self._llm if parse_file_using_llm else None)
        self._graph = Neo4jGraph()
        self._graph_builder = GraphBuilder(self._graph)
        self._knowledge_service = KnowledgeService(self._graph, self._embedding, self._llm)

    def received_file_task(self, file_task: FileTask):
        """
        由 RabbitMQ 觸發, 收到新文件時執行
        """
        file_task: FileTask = db_session.query(FileTask).get(file_task.id)
        if file_task is None:
            raise Exception("File task not found")
        # 狀態改為處理中
        file_task.status = FileTask.FileStatus.GRAPH_PROCESSING
        db_session.commit()
        # 檔案轉 Simple Graph, Document -> Parent -> Chunk
        simple_graph = self._process_file_to_graph(file_task.file_path, 
                                                   document_additional_properties={'file_task_id': file_task.id})
        simple_graph.file_id = file_task.id
        # 儲存基本圖
        self._graph_builder.save_simple_graph_to_neo4j(simple_graph.details)
        # 將檔案儲存至 Minio 並刪除本地檔案
        minio_service.upload_user_uploaded_file_to_minio(file_task)
        # 儲存simple_graph至minio
        minio_service.upload_user_uploaded_metadata_to_minio(file_task.minio_dir, simple_graph.to_dict(), MinioService.USER_UPLOADED_METADATA_TYPE.SIMPLE_GRAPH)
        # 狀態改為等待總結
        file_task.status = FileTask.FileStatus.GRAPH_ENTITY_PEDING
        db_session.commit() 
        # 準備發布下個 queue 任務
        queue_task = QueueTaskDict.create_queue_task(task_type=QueueTaskDict.TaskType.ENTITY_BUILD, msg=json.dumps(simple_graph.to_dict()))
        publish_queue_message_sync(queue_task)
        
    def _process_file_to_graph(self, file_path: str, read_file_kwargs = None, document_additional_properties=None) -> SimpleGraph:
        """
        讀取檔案內容
        轉 Simple Graph, 建立節點: Document -> Parent -> Chunk
        """
        if document_additional_properties is None:
            document_additional_properties = {}
        if read_file_kwargs is None:
            read_file_kwargs = {}

        # 開始讀取檔案
        pages = self._file_service.read_file(file_path, load_kwargs=read_file_kwargs)

        # 建立基本圖
        # simple_graph: SimpleGraph = self._graph_builder.build_chunk_graph_with_parent_child([pages], document_additional_properties=document_additional_properties)
        
        return None
    
    def received_entity_task(self, simple_graph: SimpleGraph):
        """
        由 RabbitMQ 觸發, 新文件完成 SimpleGraph 後執行
        從 Chunk 產生 Entity, 並且建立 Relationship
        """
        file_task: FileTask = db_session.query(FileTask).get(simple_graph.file_id)
        if file_task is None:
            raise Exception("File task not found")
        # 狀態改為處理中
        file_task.status = FileTask.FileStatus.GRPAH_ENTITY
        db_session.commit() 
        # 開始建立實體圖 -> 需要花大量時間 Document -> Parent -> [Chunk -> Relationship -> Entity]
        graph_documents = self._graph_builder.get_entities_graph_from_llm(self._llm, simple_graph.chunks, allowedNodes=[], allowedRelationship=[])
        self._save_entity_graph_to_neo4j(graph_documents)
        # 儲存至Minio
        entity_graph_list = [graph_document_to_dict(graph_document) for graph_document in graph_documents]
        minio_service.upload_user_uploaded_metadata_to_minio(file_task.minio_dir, entity_graph_list, MinioService.USER_UPLOADED_METADATA_TYPE.ENTITY_GRAPH_LIST)
        # 狀態改為等待總結
        file_task.status = FileTask.FileStatus.REFINED_PENDING
        db_session.commit() 
        
    def _save_entity_graph_to_neo4j(self, graph_documents: list[GraphDocument]):
        """
        儲存 Entity 到 Neo4j, 並且建立 relationship
        """
        # 儲存實體圖
        self._graph.add_graph_documents(graph_documents, baseEntityLabel=True) # 此次加入的 node 會有 __entity__ 的 label
        chunks_and_graph_documents_list = self._graph_builder.get_chunk_and_graph_document(graph_documents)

        # 以 Cypher 語法將 Chunk 與 Entity 建立 Relationship
        self._graph_builder.merge_relationship_between_chunk_and_entites(chunks_and_graph_documents_list)
        
    def received_refine_task(self, task: QueueTaskDict):
        """
        由 RabbitMQ 觸發, 批次執行
        應對新文件已創建 Entity, 需要重整 Entity 與加入建立 Community
        因此如果文件未更新應不需動作
        """
        files = db_session.query(FileTask).filter(
                        FileTask.status.in_([FileTask.FileStatus.REFINED_PENDING, FileTask.FileStatus.REFINING_KNOWLEDGE, 
                                             FileTask.FileStatus.SUMMARIZING, FileTask.FileStatus.COMPLETED]),
                        FileTask.user_operate == None
                    ).all()
        user_delete_files = db_session.query(FileTask).filter_by(user_operate=FileTask.UserOperate.DELETE).all()
        # 狀態改為重整中
        for file in files:
            file.status = FileTask.FileStatus.REFINING_KNOWLEDGE
        db_session.commit()
        # 重整知識, 刪除所有實體與社群
        self._knowledge_service.remove_all_entities_and_commnuities()
        # 取得之前已經建好的實體圖, 並建立在Neo4j
        file_graph_dict: dict[FileTask, tuple[SimpleGraph, list[GraphDocument]]] = \
                        self._knowledge_service.get_builded_task_graph(files)
        for file, (_, graph_documents) in file_graph_dict.items():
            self._save_entity_graph_to_neo4j(graph_documents)
        # 嘗試抓去過去最新的重整備份資料 (假如第一次執行也有可能沒有)
        cached_commnuities_info, cached_summaries, cached_duplicate_nodes = minio_service.download_latest_refined_data()
        # 定義重複Entity並合併
        defined_duplicate_entities = self._determine_similar_nodes_with_cached_llm(cached_duplicate_nodes)
        self._merge_nodes(defined_duplicate_entities)
        # 儲存至minio備份 - defined_duplicate_entities
        minio_service.upload_refined_metadata_to_minio(task, defined_duplicate_entities, MinioService.REFINE_METADATA_TYPE.DUPLICATE_NODES)
        # 狀態改為總結中
        for file in files:
            file.status = FileTask.FileStatus.SUMMARIZING
        db_session.commit()
        # 計算並建立社群 Entity -> [Commnuity]
        communities_info: list[Neo4jCommunityInfoDict] = self._knowledge_service.build_community_from_neo4j()
         # 儲存至minio備份 - communities_info
        minio_service.upload_refined_metadata_to_minio(task, communities_info, MinioService.REFINE_METADATA_TYPE.COMMUNITIES_INFO)
        # 開始進行總結, 嘗試抓上一個版本進行比對, 盡量避免重複計算
        summaries: list[SummaryInfoDict] = self._knowledge_service.summarize_commnuities_with_cached(communities_info,
                                                                                                     cached_commnuities_info,
                                                                                                     cached_summaries)
        self._knowledge_service.save_summary(summaries)
        # 儲存至minio備份 - summaries
        minio_service.upload_refined_metadata_to_minio(task, summaries, MinioService.REFINE_METADATA_TYPE.SUMMARIES)
        # 狀態改為完成
        for file in files:
            file.status = FileTask.FileStatus.COMPLETED
        db_session.commit()
        # 設置MSSQL已刪除紀錄為完整刪除
        for del_file in user_delete_files:
            del_file.status = FileTask.FileStatus.DELETED
            del_file.user_operate = None
        db_session.commit()
        # 備份DB
        neo4j_all_data = backup_neo4j_to_dict(self._graph)
        minio_service.upload_neo4j_backup_to_minio(neo4j_all_data)
        
    def _determine_similar_nodes_with_cached_llm(self, cached_duplicate_nodes: list[DuplicateInfoDict] = None) -> list[DuplicateInfoDict]:
        if cached_duplicate_nodes is None:
            cached_duplicate_nodes = []
        # 找出可能相似的節點 (使用 embedding + neo4j)
        similar_nodes: list[KnowledgeService.PotentialDuplicateNodeDict] = self._knowledge_service.findout_similar_nodes_rule_base()
        # 使用LLM確認是否真的為重複節點
        duplicate_nodes: list[DuplicateInfoDict] = self._knowledge_service.determine_similar_nodes_with_cached_llm(similar_nodes, cached_duplicate_nodes)
        return duplicate_nodes
    
    def _merge_nodes(self, duplicate_nodes_info: list[DuplicateInfoDict]):
        duplicate_nodes: list[list[str]] = []
        for duplicate_node in duplicate_nodes_info:
            duplicate_nodes.extend(duplicate_node["output"])
        # 進行合併節點
        self._knowledge_service.merge_nodes(duplicate_nodes)
        # 修改擁有多個desction的Entity描述, 合併為字串
        self._knowledge_service.combine_description_list()
        
    def received_restore_neo4j(self, date: str):
        db_all_data = minio_service.download_neo4j_backup_to_dict(date)
        self._knowledge_service.remove_all_data()
        restore_neo4j_from_dict(self._graph, db_all_data)
        
    def received_backup_neo4j(self):
        neo4j_all_data = backup_neo4j_to_dict(self._graph)
        minio_service.upload_neo4j_backup_to_minio(neo4j_all_data)
        
        
dataflow_manager_instance = DataflowService(parse_file_using_llm = True)


if __name__ == "__main__":
    '''
    該範例示範了如何使用 DataflowService 來進行文件處理、實體圖建立、合併節點、總結知識、刪除知識、重建知識
    其過程都支援 序列化(Json)與反序列化，並且透過 Neo4j 儲存
    若有需要自行開啟註解部分(需要 Azure LLM), 或直接使用儲存好的Json進行本地測試, 
    測試資料為完整 台灣人壽新住院醫療保險附約 檔案內容
    '''
    CUR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
    ROOT_SAVE_DIR = f'{CUR_FILE_PATH}/../test/test_data/serialization'
    test_dataflow_manager_instance = DataflowService(parse_file_using_llm = True)
    # with DataflowService(parse_file_using_llm = True) as test_dataflow_manager_instance:
    test_dataflow_manager_instance._knowledge_service.remove_all_data()
    # --------------------------------------------------------------------------
    # 需要LLM: 讀取檔案建立 SimpleGraph, 並序列化
    # file_path = os.path.join(CUR_FILE_PATH, '..', 'test', 'test_data', '台灣人壽新住院醫療保險附約.pdf')
    # read_file_kwargs = {'pages': [1, 2]} # 如果只想讀取特定幾頁可以用 read_file_kwargs={'pages': [1, 2]}
    # simple_graph: SimpleGraph = test_dataflow_manager_instance._process_file_to_graph(file_path, read_file_kwargs=read_file_kwargs)
    # with open(f'{ROOT_SAVE_DIR}/simple_graph.json', 'w') as file:
    #     json.dump(simple_graph.to_dict(), file, indent=4, ensure_ascii=False, sort_keys=True)
    # --------------------------------------------------------------------------
    # 本地測試: 讀取 SimpleGraph 反序列化, 並建立簡易圖, 並儲存至 Neo4j
    with open(f'{ROOT_SAVE_DIR}/simple_graph.json', 'r', encoding='utf-8') as file:
        simple_graph: SimpleGraph = SimpleGraph.from_dict(json.load(file))
    test_dataflow_manager_instance._graph_builder.save_simple_graph_to_neo4j(simple_graph.details)
    # --------------------------------------------------------------------------
    # 需要LLM: 讀取 SimpleGraph 反序列化, 並透過 LLM 建立實體圖, 序列化實體圖 Json
    # with open(f'{ROOT_SAVE_DIR}/simple_graph.json', 'r', encoding='utf-8') as file:
    #     simple_graph: SimpleGraph = SimpleGraph.from_dict(json.load(file))
    # graph_documents = test_dataflow_manager_instance._graph_builder.get_entities_graph_from_llm(test_dataflow_manager_instance._llm, simple_graph.chunks, 
    #                                                                                     allowedNodes=[], allowedRelationship=[])
    # entity_graph_list = [graph_document_to_dict(graph_document) for graph_document in graph_documents]
    # with open(f'{ROOT_SAVE_DIR}/entity_graph_list.json', 'w', encoding='utf-8') as file:
    #     json.dump(entity_graph_list, file, indent=4, ensure_ascii=False, sort_keys=True)
    # --------------------------------------------------------------------------
    # 本地測試: 讀取實體圖反序列化, 並儲存至 Neo4j, 並且與簡易圖合併
    with open(f'{ROOT_SAVE_DIR}/entity_graph_list.json', 'r', encoding='utf-8') as file:
        entity_graph_list = json.load(file)
    entity_graph_list = [dict_to_graph_document(graph_document) for graph_document in entity_graph_list]
    test_dataflow_manager_instance._save_entity_graph_to_neo4j(entity_graph_list)
    # --------------------------------------------------------------------------
    # 需要LLM: 找尋相似的節點, 並序列化
    # with open(f'{ROOT_SAVE_DIR}/duplicate_nodes.json', 'r', encoding='utf-8') as file:
    #     duplicate_nodes = json.load(file)
    # duplicate_nodes: list[DuplicateInfoDict] = test_dataflow_manager_instance._determine_similar_nodes_with_cached_llm(duplicate_nodes)
    # with open(f'{ROOT_SAVE_DIR}/duplicate_nodes.json', 'w', encoding='utf-8') as file:
    #     json.dump(duplicate_nodes, file, indent=4, ensure_ascii=False, sort_keys=True)
    # --------------------------------------------------------------------------
    # 本地測試: 反序列化相似節點, 並合併
    with open(f'{ROOT_SAVE_DIR}/duplicate_nodes.json', 'r', encoding='utf-8') as file:
        duplicate_nodes = json.load(file)
    test_dataflow_manager_instance._merge_nodes(duplicate_nodes)
    g1 = test_dataflow_manager_instance._knowledge_service.get_all_graph()
    # --------------------------------------------------------------------------
    # 根據結構決定是否需要LLM
    # communities_info = test_dataflow_manager_instance._knowledge_service.build_community_from_neo4j()
    # with open(f'{ROOT_SAVE_DIR}/communities_info.json', 'r', encoding='utf-8') as file:
    #     cached_communities_info = json.load(file)
    # with open(f'{ROOT_SAVE_DIR}/summaries.json', 'r', encoding='utf-8') as file:
    #     cached_summaries = json.load(file)
    # summaries = test_dataflow_manager_instance._knowledge_service.summarize_commnuities_with_cached(communities_info, cached_communities_info, cached_summaries)
    # with open(f'{ROOT_SAVE_DIR}/summaries.json', 'w', encoding='utf-8') as file:
    #     json.dump(summaries, file, indent=4, ensure_ascii=False, sort_keys=True)
    # test_dataflow_manager_instance._knowledge_service.save_summary(summaries)
    # --------------------------------------------------------------------------
    # 直接使用地端資料建立Summaries
    communities_info = test_dataflow_manager_instance._knowledge_service.build_community_from_neo4j()
    with open(f'{ROOT_SAVE_DIR}/summaries.json', 'r', encoding='utf-8') as file:
        summaries = json.load(file)
    test_dataflow_manager_instance._knowledge_service.save_summary(summaries)
    # --------------------------------------------------------------------------
    # 本地測試: 如果需要, 重建知識 並且比對, 理論上會一樣
    # test_dataflow_manager_instance._knowledge_service.remove_all_entities_and_commnuities()
    # test_dataflow_manager_instance._save_entity_graph_to_neo4j(entity_graph_list)
    # test_dataflow_manager_instance._merge_nodes(duplicate_nodes)
    
    # g2 = test_dataflow_manager_instance._knowledge_service.get_all_graph()
    # entity_diff = test_dataflow_manager_instance._knowledge_service.compare_graph_documents(g1, g2)
    # with open(f'{ROOT_SAVE_DIR}/entity_diff.json', 'w', encoding='utf-8') as file:
    #     json.dump(entity_diff, file, indent=4, ensure_ascii=False, sort_keys=True)
    # communities_info2 = test_dataflow_manager_instance._knowledge_service.build_community_from_neo4j()
    # comm_diff = compare_community_lists(communities_info1, communities_info2)
    # # duump json 
    # with open(f'{ROOT_SAVE_DIR}/comm_diff.json', 'w', encoding='utf-8') as file:
    #     json.dump(comm_diff, file, indent=4, ensure_ascii=False, sort_keys=True)
