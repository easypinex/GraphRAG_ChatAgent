from n9_a_configs import *
from n9_a_imports import *
from n9_a_functions import *

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="multiprocessing.popen_fork")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")


def main():
    # 檢查系統記憶體
    has_sufficient_memory = check_memory()
    print(f"系統記憶體檢查: {'足夠' if has_sufficient_memory else '不足'} (需要 >= 72GB)")

    # 根據記憶體大小選擇模型
    if has_sufficient_memory:
        print("使用 Qwen2-72B 模型")
        llm = Ollama(model=OLLAMA_MODEL, temperature=0.0)
        chain_type = "ollama"
    else:
        print("使用 Azure OpenAI 模型")
        llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            model_name=AZURE_OPENAI_DEPLOYMENT_NAME,
            openai_api_key=AZURE_OPENAI_API_KEY,
            temperature=0.3,
            streaming=True,
        )
        chain_type = "azure"

    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    print(f"Connecting to Redis at: {REDIS_URL}")

    prompt_template = ChatPromptTemplate.from_messages(TOPIC_SUMMARY_PROMPT)
    chain = prompt_template | llm

    start_time = time.time()
    data_frame = load_rule_pdf_to_dataframe(DEFAULT_PATH2)  # step 0: 先把資料轉pandas
    ckip = Ckip(STOP_WORDS_PATH)
    if data_frame.empty:
        sys.exit("Dataframe 處理後為空值，請確認檔案路徑與檔案是否正確")
    else:
        data_frame = content_remodified(data_frame)  # step 1: 產生 content_remodified 欄位
        content = data_frame["content_remodified"]
        data_frame = ckip_to_seglist(ckip, data_frame, content)  # step 2: 斷字斷詞拆出 seg list，並寫入dataframe
        lda_cat = LDA_Category(data_frame, TOPIC_SETTING, PASSES)
        data = lda_cat.gen_data_topic_info(SAVE_NAME2, IMG_PATH)  # step 3: 分主題
        data_topic_info = data_category_sort(data)  # step 4: 依主題topic1~ topicx排序
        print(elapsed_time("資料EDA", start_time))

        start_time = time.time()
        topics_list = topic_summary(chain, data, chain_type)  # step 5: 主題Topics總結
        topic_list_sqg = topic_lsit_segment(ckip, topics_list, data_topic_info)  # step 6: 主題Topics總結斷詞
        data_frame = content_and_topic_relation_eda(data_frame, topic_list_sqg)  # step 7: 分析 content(之後的chunk)接近哪些主題Topics

        print(elapsed_time("Topics總結", start_time))
        _ = pickle_save(SAVE_PATH + "data_rule.pkl", data_frame)
        print(f"Save Dataframe to Pickle succeeded at: {SAVE_PATH}")
        _ = pickle_save(SAVE_PATH + "data_ruletopic_info.pkl", data_topic_info)
        print(f"Save data_topic_info to Pickle succeeded at: {SAVE_PATH}")
        _ = pickle_save(SAVE_PATH + "data_ruletopics_list.pkl", topics_list)
        print(f"Save topics_list to Pickle succeeded at: {SAVE_PATH}")
        print(f"** [{SAVE_NAME2}] 資料分析與檔案建置完成 ** \n")


if __name__ == "__main__":
    main()
