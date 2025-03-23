
import logging
import json
import os
from langchain_core.prompts import ChatPromptTemplate

from ..llm_package.llm import llm_4o, llm_qwen
from ..llm_package.prompt import TOPIC_SUMMARY_PROMPT
from ..general_package.utility import check_memory, save_json
from ..ckip_package.ckip_module import ckip
from ..dto_package.chunk import Chunk
from .policy_process.policy_parser import parse_policy_content
from .underwriting_process.underwriting_parser import parse_underwriting_content


EXT = ".pdf"
LOGGER = logging.getLogger("TNPS")
CUR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
MINIO_CONTENT_PATH = f"{CUR_FILE_PATH}/../../minio_simulate/content"
MINIO_TOPIC_PATH = f"{CUR_FILE_PATH}/../../minio_simulate/topic"


def process_01_parse_content(file_dir, file_name, file_type):
    if file_type == "policy":
        chunks: list[Chunk] = parse_policy_content(file_dir, file_name)
        for chunk in chunks:
            chunk.segment_list = ckip.process_flow(chunk.content)
    elif file_type == "underwriting":
        chunks: list[Chunk] = parse_underwriting_content(file_dir, file_name)
        for chunk in chunks:
            chunk.segment_list = ckip.process_flow(chunk.summary)
    else:
        raise ValueError(f"Invalid file type: {file_type}")

    LOGGER.info(f"chunks: {json.dumps(chunks, indent=2, ensure_ascii=False, default=lambda o: o.to_dict())}")

    file_name_without_ext = file_name.replace(EXT, "")
    if file_type == "policy":
        save_json(chunks, f"{MINIO_CONTENT_PATH}/policy_doc/{file_name_without_ext}.json")
    elif file_type == "underwriting":
        save_json(chunks, f"{MINIO_CONTENT_PATH}/underwriting_doc/{file_name_without_ext}.json")

def process_02_topic_analysis():
    # 把 policy_doc 的 json 全部一起做 topic analysis
    # 把 underwriting_doc 的 json 全部一起做 topic analysis
    pass
