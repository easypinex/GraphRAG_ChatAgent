from n9_a_configs import *
from n9_a_imports import *
from n9_a_functions import *


embeddings = AzureOpenAIEmbeddings(
    model=AZURE_EMB_MODLE,
    azure_deployment=AZURE_EMB_DEPLOYMENT,
    azure_endpoint=AZURE_EMB_ENDPOINT,
    openai_api_version=AZURE_EMB_API_VERSION,
    api_key=AZURE_EMB_MODLE_API_KEY,
)

print(Confirm_EmbeddingToken_is_Working(TIKTOKEN_CACHE_DIR, CACHE_KEY, embeddings))

kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)
# 清空
_ = kg.query(DELETE_ALL_NODES)
_ = kg.query(DELETE_ALL_VECTOR_INDEX1)
_ = kg.query(DELETE_ALL_VECTOR_INDEX2)
nodes_count = kg.query(CHECK_NODE_CLEANED)
print(f"*** 目前資料庫node數為: {nodes_count} *** \n")


"""=========================== 建立條款 Nodes ==========================="""
data_frame = pickle_read(SAVE_PATH + "data.pkl")
topics_list = pickle_read(SAVE_PATH + "data_topics_list.pkl")

create_Node_Product(kg, data_frame)
create_Node_Topics(kg, topics_list)
create_Node_Chunks(kg, data_frame)
create_Relation_Tpoic_Chunks(kg)
create_Relation_Product_Chunks(kg)
create_VecotrIndex_content(kg, embeddings)


"""===========================建立投保規則 Nodes ==========================="""
data_frame = pickle_read(SAVE_PATH + "data_rule.pkl")
topics_list = pickle_read(SAVE_PATH + "data_ruletopics_list.pkl")

create_Node_RuleTopics(kg, topics_list)
create_Node_PageTable(kg, data_frame)
create_Relation_RuleTpoic_Pagetable(kg)
create_Relation_Product_Pagetable(kg)
create_VecotrIndex_pagetable(kg, embeddings)

print("** [Nodes] 建置完成 ** \n")

# 關閉連接
kg._driver.close()