from n9_a_configs import *
from n9_a_imports import *
from n9_a_functions import *

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning, module='multiprocessing.popen_fork')
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib.font_manager')

load_dotenv()
llm_stream = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    model_name=AZURE_OPENAI_DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_API_KEY,
    temperature=0.3,
    streaming=True,
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
print(f"Connecting to Redis at: {REDIS_URL}")

ollama_llm = Ollama(model=OLLAMA_MODEL ,temperature =0.0)
prompt_template = ChatPromptTemplate.from_messages(TOPIC_SUMMARY_PROMPT)
ollama_prompt = prompt_template | ollama_llm


start_time = time.time()
data_frame = load_pdf_to_dataframe(DEFAULT_PATH)              #  step 0: 先把資料轉pandas
ckip = Ckip(STOP_WORDS_PATH)
if  data_frame.empty:
    sys.exit("Dataframe 處理後為空值，請確認檔案路徑與檔案是否正確")
else: 
    data_frame = summary_manual_rule(data_frame)                                    # step 1: 產生 summary 欄位 
    content = data_frame["summary"]
    data_frame = ckip_to_seglist(ckip, data_frame, content)                         # step 2: 斷字斷詞拆出 seg list，並寫入dataframe
    lda_cat = LDA_Category(data_frame,TOPIC_SETTING, PASSES)        
    data =  lda_cat.gen_data_topic_info(SAVE_NAME,IMG_PATH)                         # step 3: 分主題
    data_topic_info = data_category_sort(data)                                      # step 4: 依主題topic1~ topicx排序
    print(elapsed_time("資料EDA", start_time))
    
    start_time = time.time()
    topics_list = topic_summary(ollama_prompt, data) # step 5: 主題Topics總結  
    topic_list_sqg = topic_lsit_segment(ckip,topics_list, data_topic_info)          # step 6: 主題Topics總結斷詞
    data_frame = content_and_topic_relation_eda(data_frame, topic_list_sqg)         # step 7: 分析 content(之後的chunk)接近哪些主題Topics
    
    print(elapsed_time("Topics總結", start_time)) 
    _ = pickle_save(SAVE_PATH + "data.pkl" ,data_frame)
    print("Save Dataframe to Pickle succeeded at :{} ".format(SAVE_PATH))
    _ = pickle_save(SAVE_PATH + "data_topic_info.pkl", data_topic_info)
    print("Save data_topic_info to Pickle succeeded at :{} ".format(SAVE_PATH))
    _ = pickle_save(SAVE_PATH + "data_topics_list.pkl", topics_list)
    print("Save topics_list to Pickle succeeded at :{} ".format(SAVE_PATH))
    print("** [{}] 資料分析與檔案建置完成 ** \n".format(SAVE_NAME))