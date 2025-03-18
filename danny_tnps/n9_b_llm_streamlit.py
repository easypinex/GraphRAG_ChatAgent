"""
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
streamlit run n9_b_llm_streamlit.py
 python -m nvitop
"""
from n9_b_imports import *
from n9_b_functions import *
from n9_b_configs import *

warnings.filterwarnings("ignore")
load_dotenv()

# REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_container_storage = redis.Redis(host='localhost', port=6379, db=0) # Redis Container 要記得先起來.....

product = redis_container_storage.lrange('product', 0, -1) 
if product ==[]:
    _ = redis_container_storage.rpush('product', "default")


# 讀取條款分類: 
with open('./pickles/data_topic_info.pkl', 'rb') as file:
    data_topic_info = pickle.load(file)

# 讀取規則分類: 
with open('./pickles/data_ruletopic_info.pkl', 'rb') as file:
    data_ruletopic_info = pickle.load(file)

#離線載入 tiktoken : https://blog.csdn.net/qq_35054222/article/details/137127660
blobpath = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
tiktoken_cache_dir = "/home/u004134/TestFolder/LLM_Soft/.tiktoken"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
assert os.path.exists(os.path.join(tiktoken_cache_dir, cache_key))


embeddings = AzureOpenAIEmbeddings(
    model =AZURE_EMB_MODLE ,
    azure_deployment=AZURE_EMB_DEPLOYMENT,
    azure_endpoint = AZURE_EMB_ENDPOINT,
    openai_api_version=AZURE_EMB_API_VERSION,
    api_key=AZURE_EMB_MODLE_API_KEY
)

llm_stream = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    model_name=AZURE_OPENAI_DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_API_KEY,
    temperature=0.3,
    streaming=True,
)
kg = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)


results = kg.query(QUERY_GET_FILENAME)
json_results_product = json.dumps(results, ensure_ascii=False, indent=4)


results = kg.query(QUERY_GET_TOPICS)
json_results_topic = json.dumps(results, ensure_ascii=False, indent=4)


results = kg.query(QUERY_GET_RULETOPICS)
json_results_ruletopic = json.dumps(results, ensure_ascii=False, indent=4)


former_llm_recommand_product = []
former_llm_recommand_topics = []

query_info_by_chunk = [VECTOR_SEARCH_BY_CHUNK, 'emb_index', 'Chunk']
query_info_by_pagetable = [VECTOR_SEARCH_BY_PAGETABLE, 'emb_index_rule', 'PageTable']
query_info_by_chunk_Blanketsearch = [VECTOR_SEARCH_BY_CHUNK_BLANKETSEARCH , 'emb_index', 'Chunk']

# history = RedisChatMessageHistory(session_id=ID, redis_url=REDIS_URL)

prompt = ChatPromptTemplate.from_messages( LLM_RAG_PROMPT)
chain = prompt | llm_stream


if "history" not in st.session_state:
    st.session_state.history = [SystemMessage(content="")]

AIMessage_temp = []
def yield_func(query_feed_to_llm, qution):
    for words in chain.stream({'refrence': query_feed_to_llm , 'input':qution}):
        time.sleep(0.01)
        AIMessage_temp.append(words.content)
        yield words.content

st.title('Chat Model: {}'.format(AZURE_OPENAI_DEPLOYMENT_NAME))
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if question := st.chat_input("What is up?"):
    st.session_state.history.append(HumanMessage(question))
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)

        query_info_by_xxx = [query_info_by_chunk, query_info_by_pagetable, query_info_by_chunk_Blanketsearch]
        json_results = [json_results_product, json_results_topic, json_results_ruletopic]
        data_topic_info_xxx = [data_topic_info, data_ruletopic_info]

        query_result_by_chunk, query_result_by_pagetable = ask_from_neo4j(  redis_container_storage, \
                                                                            llm_stream, \
                                                                            kg, \
                                                                            embeddings,\
                                                                            question,\
                                                                            query_info_by_xxx, \
                                                                            json_results,\
                                                                            data_topic_info_xxx,\
                                                                            RESPONSETHREDHOLD  
                                                                            )

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        query_feed_to_llm =[] 
        query_feed_to_llm += query_result_by_chunk
        query_feed_to_llm += query_result_by_pagetable


        print("\n ========== Query_result_by_chunk ========")
        print(query_result_by_chunk)
        print("\n")
        print("\n ========== Query_result_by_pagetable ========")
        print(query_result_by_pagetable)
        print("\n")

        response = st.write_stream(yield_func(str(query_feed_to_llm), st.session_state.history))
        AIMessage_temp = ''.join([item for item in AIMessage_temp if item])
        st.session_state.history.append(AIMessage(AIMessage_temp))

        redis_product_list = redis_container_storage.lrange('product', 0, -1) 
        if len(redis_product_list) > 1 :
            if redis_product_list[-1]!= redis_product_list[-2]:
                st.session_state.history = []
                last_n_elements = [element.decode('utf-8') for element in redis_container_storage.lrange('product', -1, -1)][0]
                _ = redis_container_storage.flushdb()
                _ = redis_container_storage.rpush('product', last_n_elements)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})






























































