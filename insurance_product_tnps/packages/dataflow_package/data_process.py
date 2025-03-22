
import logging

from langchain_core.prompts import ChatPromptTemplate

from ..llm_package.llm import llm_4o, llm_qwen
from ..llm_package.prompt import TOPIC_SUMMARY_PROMPT
from ..general_package.utility import check_memory
from ..ckip_package.ckip_module import ckip

logger = logging.getLogger("TNPS")


def process_01_parse_content(file_path, file_type):
    if file_type == "policy":
        pass
    elif file_type == "underwriting":
        pass
    else:
        raise ValueError(f"Invalid file type: {file_type}")
    
    
