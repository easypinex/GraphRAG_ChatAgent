"""
[客製設定]: 依照執行者位置目錄調整
"""
STOP_WORDS_PATH = "./ckip_model/stopwords_TW.txt"
DEFAULT_PATH = "./pdf/tnps/"
DEFAULT_PATH2 = "./pdf/tnps/rules/"
SAVE_PATH = './pickles/'
IMG_PATH = './html'
# Embedding 離線 token 準備
# 離線載入 tiktoken : https://blog.csdn.net/qq_35054222/article/details/137127660
# # blobpath = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
# # cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
# # print(cache_key) # 9b5ad71b2ce5302211f9c61530b329a4922fc6a4
CACHE_KEY = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
TIKTOKEN_CACHE_DIR = "./.tiktoken"
SAVE_NAME = "條款"
SAVE_NAME2 = "投保規則"


"""
[固定設定]
"""
AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_API_VERSION = "2024-10-01-preview"
AZURE_OPENAI_ENDPOINT = "https://sales-chatbot-llm.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "GPT4o" 

AZURE_EMB_MODLE_API_KEY = ""
AZURE_EMB_API_VERSION = "2024-02-01"
AZURE_EMB_ENDPOINT = "https://sales-chatbot-llm.openai.azure.com/"
AZURE_EMB_MODLE = 'text-embedding-3-small'
AZURE_EMB_DEPLOYMENT = 'text-embedding-3-small'

NEO4J_URI = "neo4j://localhost:7688"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "neo4j"

PASSES = 10         #30  是訓練過程中使用語料庫的次數
TOPIC_SETTING = 20  #20
NUM_TOPICS = 0      #    下面計算出來後設定 #20
PHRASE_LENGTH = 2

OLLAMA_MODEL = "qwen2:72b-instruct-q8_0"
TOPIC_SUMMARY_PROMPT = [
    ('system', '依照保險公司所提供保險商品條款的關鍵字，給我一個重點總結，只能以繁體中文回答'),
    ('system', """提供以下範例: 
                  input: [費率,高齡,非投資型,風險,評估表,說明書,審閱,調查,匯率,集體,適合度,彙繳,錄音,告知書,收付,量表,聲明書,....]
                  回應:'商品條款的重點包括.......'
                  input: [累算,財務,理賠,限制,窩心,動機,體況,做為,\n55,\n目,能力,\n○ ○,所得,彈性,情形,第1,計劃,險費,附約\n,....]
                  回應:'商品條款的重點包括.......'
                """),
    ('user', 'input: \n\n {input}')]

TOPIC_SUMMARY_PROMPT_BY_RULE = [
    ('system', '依照保險公司所提供保險商品投保規則的關鍵字，給我一個重點總結'),
    ('system', '關鍵字的內容是投保規則的敘述重點或是保險商品名稱'),
     ('system', """提供以下範例: 
                  input: [費率,高齡,非投資型,風險,評估表,說明書,審閱,調查,匯率,集體,適合度,彙繳,錄音,告知書,收付,量表,聲明書,....]
                  回應:'保險商品投保規則的重點包括.......'
                  input: [累算,財務,理賠,限制,窩心,動機,體況,做為,\n55,\n目,能力,\n○ ○,所得,彈性,情形,第1,計劃,險費,附約\n,....]
                  回應:'保險商品投保規則的重點包括.......'
                """),
    ('user', 'input: \n\n {input}'),
]


"""
        [Neo4j 操作]
"""
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
CHECK_NODE_CLEANED ="""
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
        CreateChunks.topics = $topics,
        CreateChunks.summary = $summary
RETURN CreateChunks
"""

# 創建主題與內容塊之間的關係
RELATION_QUERY_TC = """
MATCH (n)
WHERE n.topics IS NOT NULL
UNWIND n.topics AS topic
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
MERGE(CreateRuleTopics:RuleTopic {ruletopic: $ruletopic})
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
        CreatePageTable.ruletopics = $topics,
        CreatePageTable.summary = $summary,
        CreatePageTable.page = $page
RETURN CreatePageTable
"""

# 創建投保規則主題與頁面表之間的關係
RELATION_QUERY_RTPT = """
MATCH (n:PageTable)
WHERE n.ruletopics IS NOT NULL
UNWIND n.ruletopics AS ruletopic
MATCH (b:RuleTopic {ruletopic: ruletopic})
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