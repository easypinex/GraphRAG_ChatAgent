import os
import sys

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    
from langchain.schema.runnable import Runnable
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import AzureOpenAIEmbeddings

from chat_agent_module.search_params import LocalSearchParamsDict
from neo4j_module.twlf_neo4j_vector import TwlfNeo4jVector

class FileFilterRetriever(Runnable):
    configurable = {
        "fileIds": []
    }
    def __init__(self, retriever: VectorStoreRetriever):
        self.retriever: VectorStoreRetriever = retriever

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
        
        inputs = inputs.get("inputs", {})
        fileIds = inputs.get("fileIds", [])
        if fileIds is None or len(fileIds) == 0:
            fileIds = None
        update_search_kwargs = {
            "additional_match_cypher": "MATCH (n:__Entity__)<-[:HAS_ENTITY]-(c:__Chunk__)<-[:HAS_CHILD]-(p:__Parent__)-[:PART_OF]->(d:__Document__)",
            'additional_where_cypher': "WHERE $fileIds IS NULL OR d.id IN $fileIds"
        }
        if not fileIds:
            update_search_kwargs['additional_match_cypher'] = None
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
