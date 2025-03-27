
import logging
import json
import os

from ..general_package.utility import save_json
from ..ckip_package.ckip_module import CKIP
from ..dto_package.chunk import Chunk
from .policy_process.policy_parser import parse_policy_content
from .uw_process.uw_parser import parse_uw_content
from ..lda_package.lda_module import lda_analysis
from ..neo4j_package.neo4j_module import *


EXT = ".pdf"
LOGGER = logging.getLogger("TNPS")
CUR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
MINIO_CONTENT_PATH = f"{CUR_FILE_PATH}/../../minio_simulate/content"
MINIO_TOPIC_PATH = f"{CUR_FILE_PATH}/../../minio_simulate/topic"


def process_01_parse_content(file_dir, file_name, file_type):
    if file_type == "policy":
        chunks: list[Chunk] = parse_policy_content(file_dir, file_name)
        for chunk in chunks:
            chunk.segment_list = CKIP.process_flow(chunk.content)
    elif file_type == "uw":
        chunks: list[Chunk] = parse_uw_content(file_dir, file_name)
        for chunk in chunks:
            chunk.segment_list = CKIP.process_flow(chunk.summary)
    else:
        raise ValueError(f"Invalid file type: {file_type}")

    LOGGER.info(f"chunks: {json.dumps(chunks, indent=2, ensure_ascii=False, default=lambda o: o.to_dict())}")

    file_name_without_ext = file_name.replace(EXT, "")
    if file_type == "policy":
        save_json(chunks, f"{MINIO_CONTENT_PATH}/policy_doc/{file_name_without_ext}.json")
    elif file_type == "uw":
        save_json(chunks, f"{MINIO_CONTENT_PATH}/uw_doc/{file_name_without_ext}.json")

def process_02_topic_analysis():
    # policy_doc 與 uw_doc 的 json 各自資料夾的所有 json 一起做 topic analysis
    for file_type in ["policy_doc", "uw_doc"]:
        LOGGER.info(f"{file_type} topic analysis start")
        json_files = [f for f in os.listdir(f"{MINIO_CONTENT_PATH}/{file_type}") if f.endswith(".json")]

        chunk_topic_analysis = []
        for json_file in json_files:
            with open(f"{MINIO_CONTENT_PATH}/{file_type}/{json_file}", "r", encoding="utf-8") as f:
                chunks_dict = json.load(f)
                chunks = [Chunk(**chunk) for chunk in chunks_dict]

                chunk_topic_analysis.extend(chunks)

        chunk_topic_analysis, topic_summary_dict = lda_analysis(chunk_topic_analysis)
        LOGGER.info(f"chunk_topic_analysis: {json.dumps(chunk_topic_analysis, indent=2, ensure_ascii=False, default=lambda o: o.to_dict())}")
        LOGGER.info(f"topic_summary_dict: {json.dumps(topic_summary_dict, indent=2, ensure_ascii=False, default=lambda o: o.to_dict())}")
        
        save_json(chunk_topic_analysis, f"{MINIO_TOPIC_PATH}/{file_type}/chunk_topic_analysis.json")
        save_json(topic_summary_dict, f"{MINIO_TOPIC_PATH}/{file_type}/topic_summary.json")

def process_03_build_knowledge():
    # policy neo4j graph building
    with open(f"{MINIO_TOPIC_PATH}/policy_doc/chunk_topic_analysis.json", "r", encoding="utf-8") as f:
        chunks_dict = json.load(f)
        chunk_topic_analysis = [Chunk(**chunk) for chunk in chunks_dict]

    with open(f"{MINIO_TOPIC_PATH}/policy_doc/topic_summary.json", "r", encoding="utf-8") as f:
        topic_summary_dict = json.load(f)

    clear_all_nodes()

    unique_filenames = list(set(chunk.filename for chunk in chunk_topic_analysis))  # 取得所有唯一的文件名
    create_Node_Product(unique_filenames)
    create_Node_Topic(topic_summary_dict)
    create_Node_Chunk(chunk_topic_analysis)
    create_Relation_Topic_Chunks()
    create_Relation_Product_Chunks()
    create_VecotrIndex_content()

    # uw neo4j graph building
    with open(f"{MINIO_TOPIC_PATH}/uw_doc/chunk_topic_analysis.json", "r", encoding="utf-8") as f:
        chunks_dict = json.load(f)
        chunk_topic_analysis = [Chunk(**chunk) for chunk in chunks_dict]

    with open(f"{MINIO_TOPIC_PATH}/uw_doc/topic_summary.json", "r", encoding="utf-8") as f:
        topic_summary_dict = json.load(f)

    create_Node_RuleTopics(topic_summary_dict)
    create_Node_PageTable(chunk_topic_analysis)
    create_Relation_RuleTpoic_Pagetable()
    create_Relation_Product_Pagetable()
    create_VecotrIndex_pagetable()
