import os
import sys

from langchain_openai import AzureOpenAIEmbeddings

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    
from langchain.schema.runnable import Runnable
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.documents import Document
from langchain.schema.runnable import Runnable
from langchain_core.retrievers import BaseRetriever

from database import db_session
from minio_module.minio_service import minio_service
from models.file_task import FileTask
from chat_agent_module.file_filter_retriever import FileFilterRetriever

class FileMetadataRetriever(BaseRetriever):
    retriever: FileFilterRetriever
        
    def invoke(self, inputs):
        return self._get_relevant_documents(inputs)
    
    def _get_relevant_documents(self, inputs) -> list[Document]:
        docs: list[Document] = self.retriever.invoke(inputs)
        for doc in docs:
            metadata = doc.metadata
            total_fileIds: list[int] = []
            file_datas = metadata['file_datas'] if 'file_datas' in metadata else []
            metadata['file_datas'] = file_datas
            if 'fileIds' in metadata:
                fileIds = metadata['fileIds']
                if isinstance(fileIds, list):
                    total_fileIds.extend(fileIds)
                elif isinstance(fileIds, int):
                    total_fileIds.append(fileIds)
            for fileId in total_fileIds:
                fileId = 9 # mock id
                file_task = db_session.query(FileTask).filter(FileTask.id == fileId).first()
                if file_task is not None:
                    file_data = {
                        'fileId': fileId,
                        'filename': file_task.filename,
                        'filelink': minio_service.get_file_share_link(file_task.file_path)
                    }
                    file_datas.append(file_data)
            if 'fileIds' in metadata:
                del metadata['fileIds']
        return docs
    
    def stream(self, input_data):
        result = self.invoke(input_data)
        yield result  # 將最終輸出包裝為事件
    
if __name__ == "__main__":
    from chat_agent_module.twlf_vectorstore import get_localsearch_retriever, get_baseline_retriever
    embedding = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment='text-embedding-3-small',
        openai_api_version='2023-05-15'
    )
    # local_search_retriever = get_localsearch_retriever(embedding)
    # minio_retriever = FileMetadataRetriever(local_search_retriever)
    # print(minio_retriever.invoke({"question": "台灣人壽", 'inputs': {}}))
    
    baseline_retriever = get_baseline_retriever(embedding, 0.8)
    print(baseline_retriever.invoke({"question": "主契約效力停止時，要保人不得單獨申請恢復本附約之效力", 'inputs': {}}))