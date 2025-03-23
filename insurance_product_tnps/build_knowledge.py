
import os

from packages.dataflow_package.data_process import process_01_parse_content, process_02_topic_analysis
from packages.log_package.log_package import settingLogger


CUR_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    
    logger = settingLogger()

    todo_files = [
        # (f"{CUR_FILE_DIR}/upload_files/policy_doc/", "(排版)台灣人壽龍實在住院醫療健康保險附約.pdf", "policy"),
        (f"{CUR_FILE_DIR}/upload_files/uw_doc/", "(排版)台灣人壽龍實在住院醫療健康保險附約.pdf", "uw"),
    ]
    for file_dir, file_name, file_type in todo_files:
        process_01_parse_content(file_dir, file_name, file_type)    

    process_02_topic_analysis()
