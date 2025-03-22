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
    MATCH (n)  # 匹配所有節點
    DETACH DELETE n  # 刪除所有節點及其關聯的邊
"""

# 刪除向量索引1，如果存在的話
DELETE_ALL_VECTOR_INDEX1 ="""
DROP INDEX emb_index IF EXISTS  # 刪除名為emb_index的索引，如果它存在
"""

# 刪除向量索引2，如果存在的話
DELETE_ALL_VECTOR_INDEX2 ="""
DROP INDEX emb_index_rule IF EXISTS  # 刪除名為emb_index_rule的索引，如果它存在
"""

# 檢查節點是否已被清空
CHECK_NODE_CLEANED ="""
MATCH (n)  # 匹配所有節點
RETURN COUNT(n) AS node_count  # 返回當前節點的數量
"""

# 顯示索引
SHOWINDEX = """
SHOW INDEX  # 顯示當前數據庫中的所有索引
""" 

# 創建或合併產品節點
PRODUCT_QUERY = """
MERGE(CreateProducts:Product {product: $product})  # 創建或合併一個產品節點
RETURN CreateProducts  # 返回創建的產品節點
"""

# 創建或合併主題節點
TOPIC_QUERY = """
MERGE(CreateTopics:Topic {topic: $topic})  # 創建或合併一個主題節點
    ON CREATE SET 
        CreateTopics.description = $description  # 如果創建，設置描述
RETURN CreateTopics  # 返回創建的主題節點
"""

# 創建或合併內容塊節點
CHUNK_QUERY = """
MERGE(CreateChunks:Chunk {content: $content})  # 創建或合併一個內容塊節點
    ON CREATE SET 
        CreateChunks.filename = $filename,  # 設置文件名
        CreateChunks.seg_list = $seg_list,  # 設置分詞列表
        CreateChunks.topics = $topics,  # 設置相關主題
        CreateChunks.summary = $summary  # 設置內容摘要
RETURN CreateChunks  # 返回創建的內容塊節點
"""

# 創建主題與內容塊之間的關係
RELATION_QUERY_TC = """
MATCH (n)  # 匹配所有節點
WHERE n.topics IS NOT NULL  # 確保節點有主題
UNWIND n.topics AS topic  # 展開主題列表
MATCH (b:Topic {topic: topic})  # 匹配對應的主題節點
MERGE (n)-[:HAS_TOPIC]->(b)  # 創建主題與內容塊之間的HAS_TOPIC關係
"""

# 創建產品與內容塊之間的關係
RELATION_QUERY_PC = """
MATCH (n)  # 匹配所有節點
WHERE n.filename IS NOT NULL  # 確保節點有文件名
UNWIND n.filename AS filename  # 展開文件名列表
MATCH (b:Product {product: filename})  # 匹配對應的產品節點
MERGE (n)-[:FROM]->(b)  # 創建產品與內容塊之間的FROM關係
"""

# 創建內容塊的向量索引
BUILD_VECTOR_INDEX_CONTENT = """
 CREATE VECTOR INDEX `emb_index` IF NOT EXISTS  # 如果不存在，創建名為emb_index的向量索引
 FOR (e:Chunk) ON (e.contentEmbedding)  # 對Chunk節點的contentEmbedding屬性建立索引
  OPTIONS { indexConfig: {
    `vector.dimensions`: 1536,  # 設置向量維度
    `vector.similarity_function`: 'cosine'    # 設置相似度函數為餘弦相似度
 }}
"""

# 創建或合併投保規則主題節點
RULETOPIC_QUERY = """
MERGE(CreateRuleTopics:RuleTopic {ruletopic: $ruletopic})  # 創建或合併一個投保規則主題節點
    ON CREATE SET 
        CreateRuleTopics.description = $description  # 如果創建，設置描述
RETURN CreateRuleTopics  # 返回創建的投保規則主題節點
"""

# 創建或合併頁面表節點
PAGETABLE_QUERY = """
MERGE(CreatePageTable:PageTable {content: $content})  # 創建或合併一個頁面表節點
    ON CREATE SET 
        CreatePageTable.filename = $filename,  # 設置文件名
        CreatePageTable.seg_list = $seg_list,  # 設置分詞列表
        CreatePageTable.ruletopics = $topics,  # 設置相關投保規則主題
        CreatePageTable.summary = $summary,  # 設置內容摘要
        CreatePageTable.page = $page  # 設置頁碼
RETURN CreatePageTable  # 返回創建的頁面表節點
"""

# 創建投保規則主題與頁面表之間的關係
RELATION_QUERY_RTPT = """
MATCH (n:PageTable)  # 匹配所有頁面表節點
WHERE n.ruletopics IS NOT NULL  # 確保頁面表有投保規則主題
UNWIND n.ruletopics AS ruletopic  # 展開投保規則主題列表
MATCH (b:RuleTopic {ruletopic: ruletopic})  # 匹配對應的投保規則主題節點
MERGE (n)-[:HAS_RULETOPIC]->(b)  # 創建投保規則主題與頁面表之間的HAS_RULETOPIC關係
"""

# 創建頁面表與產品之間的關係
RELATION_QUERY_PPT = """
MATCH (n:PageTable)  # 匹配所有頁面表節點
WHERE n.filename IS NOT NULL  # 確保頁面表有文件名
UNWIND n.filename AS filename  # 展開文件名列表
MATCH (b:Product {product: filename})  # 匹配對應的產品節點
MERGE (n)-[:FROM]->(b)  # 創建頁面表與產品之間的FROM關係
"""

# 創建頁面表的向量索引
BUILD_VECTOR_INDEX_PAGETABLE = """
 CREATE VECTOR INDEX `emb_index` IF NOT EXISTS  # 如果不存在，創建名為emb_index的向量索引
 FOR (e:Chunk) ON (e.contentEmbedding)  # 對Chunk節點的contentEmbedding屬性建立索引
  OPTIONS { indexConfig: {
    `vector.dimensions`: 1536,  # 設置向量維度
    `vector.similarity_function`: 'cosine'    # 設置相似度函數為餘弦相似度
 }}
"""