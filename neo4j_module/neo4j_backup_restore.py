import sys
import os
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from langchain_community.graphs import Neo4jGraph
import json
import uuid
from logger.logger import get_logger

logging = get_logger()

# 1. 備份功能：將 Neo4j 節點和關係備份為字典格式
def backup_neo4j_to_dict(graph: Neo4jGraph) -> dict:
    # 查詢所有節點，並添加唯一識別符
    nodes_query = """
    MATCH (n)
    RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties
    """
    nodes = graph.query(nodes_query)
    
    # 查詢所有關係
    relationships_query = """
    MATCH (a)-[r]->(b)
    RETURN elementId(a) AS start_node, elementId(b) AS end_node, type(r) AS type, properties(r) AS properties
    """
    relationships = graph.query(relationships_query)

    # 組合節點和關係
    backup_dict = {
        "nodes": nodes,
        "relationships": relationships
    }
    return backup_dict

# 2. 還原功能：將備份的字典資料還原到 Neo4j
def restore_neo4j_from_dict(graph: Neo4jGraph, backup_data: dict):
    # 儲存舊 ID 與新節點的唯一識別符映射
    nodes_map = {}

    # 還原節點
    for node_data in backup_data["nodes"]:
        labels = ":".join(node_data["labels"])  # 標籤
        properties = {**node_data["properties"]}  # 節點屬性
        
        # 添加唯一標識符
        uuid_value = str(uuid.uuid4())
        properties['uuid'] = uuid_value

        # 創建節點
        create_node_query = f"""
        CREATE (n:{labels} $props)
        RETURN elementId(n) AS new_id
        """
        result = graph.query(create_node_query, params={"props": properties})
        nodes_map[node_data["id"]] = uuid_value

    # 還原關係
    for rel_data in backup_data["relationships"]:
        start_uuid = nodes_map[rel_data["start_node"]]
        end_uuid = nodes_map[rel_data["end_node"]]
        rel_type = rel_data["type"]
        properties = rel_data["properties"]

        # 使用唯一識別符匹配節點並創建關係
        create_relationship_query = f"""
        MATCH (a {{uuid: '{start_uuid}'}}), (b {{uuid: '{end_uuid}'}})
        CREATE (a)-[r:{rel_type} $props]->(b)
        """
        graph.query(create_relationship_query, params={"props": properties})

    logging.info("Data successfully restored to Neo4j without using ID().")

# 3. 主程式：備份與還原邏輯
def main():
    # 初始化 Neo4jGraph 連接
    graph = Neo4jGraph(
        url="bolt://localhost:7687", 
        username="neo4j", 
        password="2wsx3edc"
    )
    backup_file = "neo4j_backup.json"
    
    # 選擇操作：備份或還原
    action = input("Enter 'backup' to backup data or 'restore' to restore data: ").strip().lower()

    if action == "backup":
        # 執行備份
        backup_data = backup_neo4j_to_dict(graph)
        # 將備份資料存成 JSON 文件
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(backup_data, f, ensure_ascii=False)
        logging.info(f"Data successfully backed up to {backup_file}")
    elif action == "restore":
        # 讀取備份文件
        with open(backup_file, "r", encoding="utf-8") as f:
            backup_data = json.load(f)
        # 執行還原
        restore_neo4j_from_dict(graph, backup_data)
    else:
        logging.error("Invalid action. Please enter 'backup' or 'restore'.")

if __name__ == "__main__":
    main()
