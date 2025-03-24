
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
from .uw_process.uw_parser import parse_uw_content
from ..lda_package.lda_module import lda_analysis


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
    elif file_type == "uw":
        chunks: list[Chunk] = parse_uw_content(file_dir, file_name)
        for chunk in chunks:
            chunk.segment_list = ckip.process_flow(chunk.summary)
    else:
        raise ValueError(f"Invalid file type: {file_type}")

    LOGGER.info(f"chunks: {json.dumps(chunks, indent=2, ensure_ascii=False, default=lambda o: o.to_dict())}")

    file_name_without_ext = file_name.replace(EXT, "")
    if file_type == "policy":
        save_json(chunks, f"{MINIO_CONTENT_PATH}/policy_doc/{file_name_without_ext}.json")
    elif file_type == "uw":
        save_json(chunks, f"{MINIO_CONTENT_PATH}/uw_doc/{file_name_without_ext}.json")

def process_02_topic_analysis():
    # 把 policy_doc 的 json 全部一起做 topic analysis
    json_files = [f for f in os.listdir(f"{MINIO_CONTENT_PATH}/policy_doc") if f.endswith(".json")]

    all_file_chunks = []
    for json_file in json_files:
        with open(f"{MINIO_CONTENT_PATH}/policy_doc/{json_file}", "r", encoding="utf-8") as f:
            chunks_dict = json.load(f)
            chunks = [Chunk(**chunk) for chunk in chunks_dict]

            all_file_chunks.extend(chunks)

    all_file_chunks, topic_summary_dict = lda_analysis(all_file_chunks)
    LOGGER.info(f"all_file_chunks: {json.dumps(all_file_chunks, indent=2, ensure_ascii=False, default=lambda o: o.to_dict())}")
    
    save_json(all_file_chunks, f"{MINIO_TOPIC_PATH}/policy_doc/topic_analysis.json")
    save_json(topic_summary_dict, f"{MINIO_TOPIC_PATH}/policy_doc/topic_summary.json")
    # 把 uw_doc 的 json 全部一起做 topic analysis
    pass
