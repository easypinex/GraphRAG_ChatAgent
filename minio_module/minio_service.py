from enum import Enum
import sys
import os
import ssl
import urllib3

from dataflow_module.rabbitmq_task import QueueTaskDict
from graph_module.dto.comnuity_info_dict import Neo4jCommunityInfoDict
from graph_module.dto.summary_info_dict import SummaryInfoDict
from models.file_task import FileTask

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from minio_module.minio_client import MinioClient
from datetime import datetime

from database import db_session

class MinioService:
    class USER_UPLOADED_METADATA_TYPE(str, Enum):
        SIMPLE_GRAPH = "simple_graph"
        ENTITY_GRAPH_LIST = "entity_graph_list"
        
    class REFINE_METADATA_TYPE(str, Enum):
        DUPLICATE_NODES = "duplicate_nodes"
        SUMMARIES = "summaries"
        COMMUNITIES_INFO = "communities_info"
        
    def __init__(self):
        cert_verify = os.getenv("MINIO_CERT_VERIFY", "true").lower() == "true"
        http_client = None
        secure = os.environ["MINIO_SECURE"].lower() == "true"
        if secure and not cert_verify:
            http_client = urllib3.PoolManager(
                cert_reqs='CERT_NONE',  # 不要求憑證
                assert_hostname=False    # 不檢查主機名稱
            )
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self._mini_client = MinioClient(endpoint=os.environ["MINIO_SERVER"], 
                                        access_key=os.environ["MINIO_ACCESS_KEY"], 
                                        secret_key=os.environ["MINIO_SECRET_KEY"], 
                                        secure=secure,
                                        http_client=http_client,
                                        )
        self._bucket = os.environ["MINIO_BUCKET"]
        self._floder = "user_uploaded_file"
        
    def upload_user_uploaded_file_to_minio(self, filetask: FileTask):
        # {floder}/{date}_{time}_{file_id}_{file_name}/{filename}
        # like user_uploaded_file/20241217_1420_1_filename/file_name
        now_str = datetime.now().strftime("%Y%m%d_%H%M")
        minio_destnation_dir = f"{self._floder}/{now_str}_{filetask.id}_{filetask.filename}"
        filetask.minio_dir = minio_destnation_dir
        minio_destnation_path = f"{minio_destnation_dir}/{filetask.filename}"
        upload_result = self._mini_client.upload_file_to_minio(self._bucket, filetask.file_path, minio_destnation_path)
        if upload_result:
            os.remove(filetask.file_path)
            filetask.filedir = None
            db_session.commit()
        
    def upload_user_uploaded_metadata_to_minio(self, minio_dir: str, metadata: dict, meata_type: USER_UPLOADED_METADATA_TYPE):
        minio_destnation_path = f"{minio_dir}/{meata_type.value}.json"
        self._mini_client.upload_dict_as_json(self._bucket, metadata, minio_destnation_path)
        
    def download_user_uploaded_metadata_from_minio_as_dict(self, filetask: FileTask, meata_type: USER_UPLOADED_METADATA_TYPE) -> dict:
        minio_destnation_path = f"{filetask.minio_dir}/{meata_type.value}.json"
        return self._mini_client.download_json_as_dict(self._bucket, minio_destnation_path)
        
    def upload_refined_metadata_to_minio(self, task: QueueTaskDict, metadata: dict|list, meata_type: REFINE_METADATA_TYPE):
        date = datetime.strptime(task['send_datetime'], "%Y%m%d_%H%M%S").strftime("%Y%m%d")
        floder = f"refine_knowledge_graph/{date}"
        self._mini_client.upload_dict_as_json(self._bucket, metadata, f"{floder}/{meata_type.value}.json")
    
    def download_refined_metadata_from_minio_as_dict(self, date: str, meata_type: REFINE_METADATA_TYPE) -> dict|list:
        floder = f"refine_knowledge_graph/{date}"
        return self._mini_client.download_json_as_dict(self._bucket, f"{floder}/{meata_type.value}.json")
        
    def upload_neo4j_backup_to_minio(self, db_data):
        date = datetime.now().strftime('%Y%m%d')
        floder = f"neo4j_backup/{date}"
        self._mini_client.upload_dict_as_json(self._bucket, db_data, f"{floder}/neo4j_dump.json")
    
    def download_neo4j_backup_to_dict(self, date: str) -> dict:
        floder = f"neo4j_backup/{date}"
        return self._mini_client.download_json_as_dict(self._bucket, f"{floder}/neo4j_dump.json")
    
    def check_neo4j_backup_file_exist(self, date: str) -> bool:
        floder = f"neo4j_backup/{date}"
        return self._mini_client.check_file_exists(self._bucket, f"{floder}/neo4j_dump.json")
    
    def download_latest_refined_data(self) -> tuple[list[Neo4jCommunityInfoDict], list[SummaryInfoDict, list[list[str]]]]:
        floder = f"refine_knowledge_graph/"
        folders: list[str] = self._mini_client.list_files(self._bucket, floder)
        if len(folders) == 0:
            return [], [], []
        folders.sort()
        communities, summaries, duplicate_nodes = None, None, None
        while communities is None or summaries is None and len(folders) > 0:
            date = folders.pop()[len(floder):].replace('/', '')
            communities = self.download_refined_metadata_from_minio_as_dict(date, self.REFINE_METADATA_TYPE.COMMUNITIES_INFO)
            summaries = self.download_refined_metadata_from_minio_as_dict(date, self.REFINE_METADATA_TYPE.SUMMARIES)
            duplicate_nodes = self.download_refined_metadata_from_minio_as_dict(date, self.REFINE_METADATA_TYPE.DUPLICATE_NODES)
        if communities is None or summaries is None or duplicate_nodes is None:
            return [], [], []
        return communities, summaries, duplicate_nodes
        
        
        
minio_service = MinioService()