import os
import sys

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    
from langchain.schema.runnable import Runnable
from langchain_core.vectorstores.base import VectorStoreRetriever

class FileFilterRetriever(Runnable):
    configurable = {
        "fileIds": []
    }
    def __init__(self, retriever: VectorStoreRetriever, node_label="__Entity__"):
        self.retriever: VectorStoreRetriever = retriever
        self.node_label = node_label

    def update_params(self, new_params):
        """動態更新 retriever 的 search_kwargs['params']"""
        if 'params' not in self.retriever.search_kwargs:
            self.retriever.search_kwargs['params'] = {}
        self.retriever.search_kwargs['params'].update(new_params)
    def update_search_kwargs(self, search_kwargs: dict):
        self.retriever.search_kwargs.update(search_kwargs)
    
    def _update_params(self, inputs):
        if not isinstance(inputs, dict) :
            raise TypeError("query must be a dict")
        missing_keys = list(set(['inputs', 'question']) - set(inputs.keys()))
        if len(missing_keys) > 0:
            raise ValueError(f"Missing keys: {', '.join(missing_keys)}")
        
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
        question = inputs.get("question")
        self._update_params(inputs)
        return self.retriever.invoke(question)
    
    async def ainvoke(self, inputs, *args, **kwargs):
        question = inputs.get("question")
        self._update_params(inputs)
        return await self.retriever.ainvoke(question)
