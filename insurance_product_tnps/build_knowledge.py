
from packages.dataflow_package.data_process import process_01_parse_content
from packages.log_package.log_package import settingLogger

if __name__ == "__main__":
    
    logger = settingLogger()

    todo_files = [
        ("/Users/derick/develop/repo/GraphRAG_ChatAgent/insurance_product_tnps/upload_files/policy_doc/(排版)台灣人壽龍實在住院醫療健康保險附約.pdf", "policy"),
        ("/Users/derick/develop/repo/GraphRAG_ChatAgent/insurance_product_tnps/upload_files/underwriting_doc/(排版)台灣人壽龍實在住院醫療健康保險附約.pdf", "underwriting"),
    ]
    for file_path, file_type in todo_files:
        process_01_parse_content(file_path, file_type)

    
