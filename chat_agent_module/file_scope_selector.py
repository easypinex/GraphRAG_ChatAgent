import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    
from typing import List, Optional
from langchain.schema.runnable import Runnable
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

from chat_agent_module.inputs_validator import inputs_validator
from database import db_session
from models.file_task import FileTask
from prompts.prompts import QUESTION_HISTORY_PROMPT, RELATED_FILE_IDS_PROMPT
from langchain_core.output_parsers import StrOutputParser

class FileScopeSelector(Runnable):
    class RelatedFileScope(BaseModel):
        fileIds: Optional[List[int]] = Field(
            description="可能有問題答案的文件清單"
        )
    '''
    根據問題, 透過llm找出相關的文件範圍
    '''
    def __init__(self, llm):
        
        self.extraction_llm = llm.with_structured_output(FileScopeSelector.RelatedFileScope)
        self.related_chain = (
            RELATED_FILE_IDS_PROMPT | self.extraction_llm
        )
        
    def invoke(self, inputs: dict, config: Optional[RunnableConfig] = None, *args, **kwargs) -> dict:
        inputs_validator(inputs)
        inside_inputs: dict = inputs.get("inputs", {})
        fileIds: list[int] = inside_inputs.get("fileIds", [])
        fileIds = list(set(fileIds)) # 去重
        inside_inputs['fileIds'] = fileIds
        question = inputs.get("question")
        if len(fileIds) > 0:
            return inputs
        inside_inputs['fileIds'] = self.get_related_file_ids(question)
        return inputs
        
        
    def get_related_file_ids(self, question: str) -> list[int]:
        all_file_tasks =  db_session.query(FileTask).filter(
                        FileTask.status.in_([FileTask.FileStatus.COMPLETED])
                    ).all()
        all_file_tasks_str = self._get_file_tasks_str(all_file_tasks)
        scope: FileScopeSelector.RelatedFileScope = self.related_chain.invoke({"question": question, "file_list": all_file_tasks_str})
        if scope is not None and scope.fileIds is not None:
            return scope.fileIds
        return []
        
    def _get_file_tasks_str(self, file_tasks: List[FileTask]) -> str:
        result = ''
        for file_task in file_tasks:
            result += f"商品ID:{file_task.id}, 商品名稱:{file_task.filename}\n"
        return result
        
if __name__ == '__main__':
    from langchain_openai import AzureChatOpenAI
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME_MAIN"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        temperature=0
    )
    selector = FileScopeSelector(llm)
    result = selector.invoke({"question": "新住院有哪些條款?", "inputs": {"question": "新住院有哪些條款?"}})
    print(result)