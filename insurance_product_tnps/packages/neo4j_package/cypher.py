# 刪除所有節點
DELETE_ALL_NODES = """
    MATCH (n)
    DETACH DELETE n
"""

# 刪除向量索引1，如果存在的話
DELETE_ALL_VECTOR_INDEX1 ="""
DROP INDEX emb_index IF EXISTS
"""

# 刪除向量索引2，如果存在的話
DELETE_ALL_VECTOR_INDEX2 ="""
DROP INDEX emb_index_rule IF EXISTS
"""

# 檢查節點是否已被清空
GET_NODE_COUNT ="""
MATCH (n)
RETURN COUNT(n) AS node_count
"""

# 顯示索引
SHOWINDEX = """
SHOW INDEX
""" 

# 創建或合併產品節點
PRODUCT_QUERY = """
MERGE(CreateProducts:Product {product: $product})
RETURN CreateProducts
"""

# 創建或合併主題節點
TOPIC_QUERY = """
MERGE(CreateTopics:Topic {topic: $topic})
    ON CREATE SET 
        CreateTopics.description = $description
RETURN CreateTopics
"""

# 創建或合併內容塊節點
CHUNK_QUERY = """
MERGE(CreateChunks:Chunk {content: $content})
    ON CREATE SET 
        CreateChunks.filename = $filename,
        CreateChunks.seg_list = $seg_list,
        CreateChunks.topic_list = $topic_list,
        CreateChunks.summary = $summary
RETURN CreateChunks
"""

# 創建主題與內容塊之間的關係
RELATION_QUERY_TC = """
MATCH (n)
WHERE n.topic_list IS NOT NULL
UNWIND n.topic_list AS topic
MATCH (b:Topic {topic: topic})
MERGE (n)-[:HAS_TOPIC]->(b)
"""

# 創建產品與內容塊之間的關係
RELATION_QUERY_PC = """
MATCH (n)
WHERE n.filename IS NOT NULL
UNWIND n.filename AS filename
MATCH (b:Product {product: filename})
MERGE (n)-[:FROM]->(b)
"""

# 創建內容塊的向量索引
BUILD_VECTOR_INDEX_CONTENT = """
 CREATE VECTOR INDEX `emb_index` IF NOT EXISTS
 FOR (e:Chunk) ON (e.contentEmbedding)
  OPTIONS { indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
 }}
"""

# 創建或合併投保規則主題節點
RULETOPIC_QUERY = """
MERGE(CreateRuleTopics:RuleTopic {topic: $topic})
    ON CREATE SET 
        CreateRuleTopics.description = $description
RETURN CreateRuleTopics
"""

# 創建或合併頁面表節點
PAGETABLE_QUERY = """
MERGE(CreatePageTable:PageTable {content: $content})
    ON CREATE SET 
        CreatePageTable.filename = $filename,
        CreatePageTable.seg_list = $seg_list,
        CreatePageTable.topic_list = $topic_list,
        CreatePageTable.summary = $summary,
        CreatePageTable.page = $page
RETURN CreatePageTable
"""

# 創建投保規則主題與頁面表之間的關係
RELATION_QUERY_RTPT = """
MATCH (n:PageTable)
WHERE n.topic_list IS NOT NULL
UNWIND n.topic_list AS topic
MATCH (b:RuleTopic {topic: topic})
MERGE (n)-[:HAS_RULETOPIC]->(b)
"""

# 創建頁面表與產品之間的關係
RELATION_QUERY_PPT = """
MATCH (n:PageTable)
WHERE n.filename IS NOT NULL
UNWIND n.filename AS filename
MATCH (b:Product {product: filename})
MERGE (n)-[:FROM]->(b)
"""

# 創建頁面表的向量索引
BUILD_VECTOR_INDEX_PAGETABLE = """
 CREATE VECTOR INDEX `emb_index` IF NOT EXISTS
 FOR (e:Chunk) ON (e.contentEmbedding)
  OPTIONS { indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
 }}
"""
