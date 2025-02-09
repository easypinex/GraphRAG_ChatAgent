import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    
from chat_agent_module.inputs_validator import inputs_validator
    
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

class FileFilterRetriever(BaseRetriever):
    retriever: VectorStoreRetriever
    node_label: str = "__Entity__"

    def update_params(self, new_params):
        """動態更新 retriever 的 search_kwargs['params']"""
        if 'params' not in self.retriever.search_kwargs:
            self.retriever.search_kwargs['params'] = {}
        self.retriever.search_kwargs['params'].update(new_params)
    def update_search_kwargs(self, search_kwargs: dict):
        self.retriever.search_kwargs.update(search_kwargs)
    
    def _update_params(self, inputs):
        inputs_validator(inputs)
        inputs: dict = inputs.get("inputs", {})
        fileIds: list[int] = inputs.get("fileIds", [])
        if fileIds is None or len(fileIds) == 0:
            fileIds = None
        update_search_kwargs = {
            'additional_where_cypher': "WHERE $fileIds IS NULL OR d.file_task_id IN $fileIds",
        }
        if self.node_label == '__Chunk__':
            match_cypher = "MATCH (n:__Chunk__)<-[:HAS_CHILD]-(p:__Parent__)-[:PART_OF]->(d:__Document__)"
            retrieval_cypher = (
                f"RETURN node.`content` AS text, score, "
                f"node {{.*, `content`: Null, "
                f"fileIds: fileTaskId, "
                f"`embedding`: Null, id: Null }} AS metadata"
            )
            update_search_kwargs['retrieval_query'] = retrieval_cypher
        elif self.node_label == '__Entity__':
            match_cypher = "MATCH (n:__Entity__)<-[:HAS_ENTITY]-(c:__Chunk__)<-[:HAS_CHILD]-(p:__Parent__)-[:PART_OF]->(d:__Document__)"

        update_search_kwargs['additional_match_cypher'] = match_cypher
        if not fileIds:
            update_search_kwargs['additional_where_cypher'] = None

        self.update_params({"fileIds": fileIds})
        self.update_search_kwargs(update_search_kwargs)
        
    def invoke(self, inputs, *args, **kwargs):
        return self._get_relevant_documents(inputs)
    
    def _get_relevant_documents(self, inputs: dict) -> list[Document]:
        question = inputs.get("question")
        self._update_params(inputs)
        return self.retriever.invoke(question)

    def stream(self, input_data):
        result = self.invoke(input_data)
        yield result  # 將最終輸出包裝為事件
        
if __name__ == "__main__":
    from chat_agent_module.twlf_vectorstore import get_localsearch_retriever, get_baseline_retriever
    from langchain_openai import AzureOpenAIEmbeddings
    embedding = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment='text-embedding-3-small',
        openai_api_version='2023-05-15'
    )
    retriever = get_baseline_retriever(embedding, 0.8)
    result = retriever.invoke({"question": "主契約效力停止時，要保人不得單獨申請恢復本附約之效力", 'inputs': {}})
    print(result)