import psutil

from n9_a_configs import *
from n9_a_imports import *

def check_memory():
    """
    檢查系統記憶體大小
    返回：
        bool: 如果記憶體 >= 72GB 返回 True，否則返回 False
    """
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)  # 轉換為 GB
    return memory_gb >= 72

# 去重複
def rm_duplicate(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
    
def policy_product_chunk_rule(pages):
    """
    此函數用於處理政策文檔的分塊規則，主要處理包含【】標記的條文內容。
    處理流程：
    1. 合併所有頁面內容
    2. 按【】標記分割條文
    3. 處理前言部分
    4. 處理附表部分
    
    示例輸入：
    pages = [
        Page(page_content="前言內容【第一條】條文1內容【第二條】條文2內容附表1"),
        Page(page_content="附表2內容")
    ]
    
    示例輸出：
    [
        "前言內容",
        "【第一條】條文1內容",
        "【第二條】條文2內容",
        "附表1",
        "附表2內容"
    ]
    """
    # 初始化一個空字串，用來儲存所有頁面的內容
    all_content = ""
    # 遍歷每一頁，將頁面的內容累加到 all_content 中
    for page in pages: 
        all_content += page.page_content
    
    # 使用正則表達式按條文拆分內容，模式為匹配以「【」開頭的條文
    # 例如：如果 all_content = "【條文1】內容1【條文2】內容2"
    # 那麼 match 將會是 ["【條文1】內容1", "【條文2】內容2"]
    pattern = r'【.*?】.*?(?=【|$)'
    match = re.findall(pattern, all_content, re.DOTALL)
    
    # 補充第一個條文的內容
    # 例如：如果 all_content = "前言【條文1】內容1【條文2】內容2"
    # 那麼 first_chunk 將會是 "前言"
    pattern = r'^(.*?)【'
    first_chunk = re.findall(pattern, all_content, re.DOTALL)[0]
    
    # 將第一個條文插入到結果的最前面
    # 例如：match 現在是 ["【條文1】內容1", "【條文2】內容2"]
    # 插入後 match 將變成 ["前言", "【條文1】內容1", "【條文2】內容2"]
    match.insert(0, first_chunk)
    
    # 補充最後一筆資料的尾巴
    # 將最後一筆資料根據「附表」進行拆分
    # 例如：如果 match[-1] = "內容2附表1"
    # 那麼 split_entries 將會是 ["內容2", "1"]
    split_entries = match[-1].split('附表')
    
    # 將拆分後的資料依序補在結果的後面
    # 例如：match 現在是 ["前言", "【條文1】內容1", "【條文2】內容2"]
    # 插入後 match 將變成 ["前言", "【條文1】內容1", "【條文2】內容2", "內容2", "附表1"]
    match = match[:-1] + [split_entries[0]] + ['附表' + entry for entry in split_entries[1:]]
    
    # 返回拆分後的結果
    return match 

def dict_to_string(data):
    result = []
    for key, value in data.items():
        for sub_key, sub_value in value.items():
            if sub_value is not None:
                title = sub_value[0] if sub_value[0] else "標題"
                content = "\n".join([item for item in sub_value[1:] if item])
                result.append(f"{title}:\n{content}\n")
    return "\n".join(result)

def generate_response_for_query(chain, query_params):
    """
    根據不同的 LLM 類型生成回應
    """
    if isinstance(chain, AzureChatOpenAI):
        # Azure OpenAI 的處理方式
        for r in chain.stream(query_params):
            yield r
    else:
        # Ollama 的處理方式
        response = chain.invoke(query_params)
        yield response

def pickle_save(SAVE_PATH, data):
    with open(SAVE_PATH, 'wb') as file:
        pickle.dump(data, file)
    return 

def pickle_read(SAVE_PATH):
    with open(SAVE_PATH, 'rb') as file:
        data = pickle.load(file)
    return data

def load_to_file_path(DEFAULT_PATH):
    file_path = []
    for filename in os.listdir(DEFAULT_PATH):
        if filename.endswith('.pdf'):
            file_path.append(filename)
    return file_path

def pdf_to_pages(FP):
    pdf = pdfplumber.open(FP) 
    pages = pdf.pages
    table_with_page = dict()
    for index, page in enumerate(pages):
        text = page.extract_tables()    # 取出文字
        if text != []:
            # print("頁數:{} --> {}".format(index+1, text))
            table_with_page[index+1] = text
    return table_with_page

def data_split(table_with_page):
    dict_page_pd = dict()
    page_list = []
    for page, data in table_with_page.items():
        df_list = []
        df = pd.DataFrame(data[:])
        for i in range(df.shape[0]):
            df_list.append(df.iloc[[i]])
            page_list.append(page)
        dict_page_pd[page] =  df_list
    return dict_page_pd, page_list
    
def pandas_to_dict_format(dict_page_pd): 
    dict_page_str = dict()
    
    for page, dfs in dict_page_pd.items():
        str_list = []
        for _, df in enumerate(dfs):
            your_json = df.to_json(force_ascii=False)
            stringformat = str(your_json)  # 如要轉回diCT : json.loads(stringformat)
            str_list.append(stringformat)
        dict_page_str[page] = str_list
    return dict_page_str

def load_pdf_to_dataframe(DEFAULT_PATH): # step 0: 先把資料轉pandas
    # 初始化一個空的列表，用來儲存所有內容
    ALL_CONTENTS = []
    # 初始化兩個空的列表，分別用來儲存檔案名稱和內容
    dict_col1 = []
    dict_col2 = []
    # 獲取指定路徑下所有PDF檔案的檔案名稱
    file_path = load_to_file_path(DEFAULT_PATH)
    
    # 遍歷每個檔案名稱
    for index, _ in enumerate(file_path):
        # 獲取當前檔案的完整路徑
        FP = file_path[index]    
        # 使用PyPDFLoader載入PDF檔案並將其分割成頁面
        loader = PyPDFLoader(DEFAULT_PATH + FP)
        pages = loader.load_and_split()
        # 對每一頁進行處理，提取出有用的內容
        ALL_CONTENTS = policy_product_chunk_rule(pages)
        # 將檔案名稱重複添加到dict_col1中，數量與ALL_CONTENTS相同
        dict_col1 += [FP] * len(ALL_CONTENTS)
        # 將提取的內容添加到dict_col2中
        dict_col2 += ALL_CONTENTS
    # 將檔案名稱和內容組合成一個字典
    data_dict = {
        'filename':dict_col1,
        'content': dict_col2}
    
    # 檢查檔案名稱和內容的長度是否一致
    if len(dict_col1) == len(dict_col2):
        # 如果一致，將字典轉換為DataFrame並新增一個summary欄位
        data_frame = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_dict.items()]))
        data_frame['summary'] = None
    else: 
        # 如果不一致，返回一個空的DataFrame並提示用戶檢查
        data_frame = pd.DataFrame()
        print("Dataframe  is Empty, Please check ...")
    return data_frame

def load_rule_pdf_to_dataframe(DEFAULT_PATH2): 
    ALL_CONTENTS = []
    dict_col1 = []
    dict_col2 = []
    dict_col3 = []
    file_path = load_to_file_path(DEFAULT_PATH2)

    for index, _ in enumerate(file_path):
        FP = file_path[index]  
        table_with_page = pdf_to_pages(DEFAULT_PATH2 + "/"+ FP)
        dict_page_pd, page_list = data_split(table_with_page)
        ALL_CONTENTS = pandas_to_dict_format(dict_page_pd)

        page__wtih_data_list=[]
        for key, value in ALL_CONTENTS.items():
            page__wtih_data_list+=(value)
        dict_col1 += [FP] * len(page__wtih_data_list)
        dict_col2 += page__wtih_data_list
        dict_col3 += page_list
    data_dict = {
        'filename':dict_col1,
        'content': dict_col2,
        'page': dict_col3
    }
    if len(dict_col1) == len(dict_col2):
        data_frame = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_dict.items()]))
        data_frame["content_remodified"] = None
        data_frame['seg_list']= None
    else: 
        data_frame = pd.DataFrame()
        print("Dataframe  is Empty, Please check ...")
    return data_frame

def summary_manual_rule(data_frame): 
    for index, str_data in enumerate(data_frame['content'].tolist()):
        pattern_articles = r"【(.*?)】"
        matches = re.findall(pattern_articles, str_data)
        if matches:
            matches = matches[0].replace("【", "")
            matches = matches.replace("】", "")
            data_frame.at[index, 'summary'] = matches
        else:
            pattern_appendix = r"^(附表.*?)(?=\n)"
            match = re.search(pattern_appendix, str_data)
            if match:
                result = match.group(1)
                data_frame.at[index, 'summary'] = result
            else:
                data_frame.at[index, 'summary'] = str_data 
    data_frame[data_frame['summary'] == '']
    data_frame[data_frame['summary'].isna()]
    return data_frame

def content_remodified(data_frame):
    data_frame["content_remodified"] = None
    for index in data_frame.index:
        v_str_final =""
        for _, value in (json.loads(data_frame["content"].iloc[index])).items():
            v_str_list = []
            for _ , v_list in value.items():
                if v_list is not None:
                    v_list = [x for x in v_list if x is not None]
                    v_str = ",".join(v_list)
                    v_str_list.append(v_str)
                    v_str = ",".join(v_str_list)
                else:
                    v_str = ""
            v_str_final+=v_str
        data_frame["content_remodified"].iloc[index] = v_str_final
    return data_frame

def ckip_to_seglist(ckip, data_frame, content): 
    data_frame['seg_list'] = None
    for i in range(len(data_frame)):
        ws_results = ckip.do_CKIP_WS(content.iloc[i])
        pos_results = ckip.do_CKIP_POS(ws_results)
        word_lst = ckip.cleaner(ws_results, pos_results)
        data_frame.at[i, 'seg_list'] = word_lst
    return data_frame 

def topic_lsit_segment(ckip,topics_list, data_topic_info):
    topics_list_seg_lst = []
    for i in range(len(topics_list)):
        ws_results = ckip.do_CKIP_WS(topics_list[i])
        pos_results = ckip.do_CKIP_POS(ws_results)
        word_lst = ckip.cleaner(ws_results, pos_results)
        topics_list_seg_lst.append(word_lst)
    
    topic_list_sqg = pd.DataFrame( {'topic_list_sqg':topics_list_seg_lst})
    topic_list_sqg["Category"] = rm_duplicate(data_topic_info["Category"].tolist())
    return topic_list_sqg

def Category_split(data):
    return int(data.split("Topic")[1])

def data_category_sort(data):
    data_topic_info = data.topic_info[["Term","Category"]]
    data_topic_info = data_topic_info[data_topic_info["Category"]!="Default"]
    data_topic_info["Category_int"] = data_topic_info["Category"].apply(Category_split)
    data_topic_info = data_topic_info.sort_values(by='Category_int')
    return data_topic_info

def topic_summary(chain, data, chain_type="azure"):
    """
    根據不同的 LLM 類型生成主題總結
    
    Args:
        chain: LLM chain
        data: 主題資料
        chain_type: LLM 類型 ("azure" 或 "ollama")
    """
    ollama_inference_count = 0
    topics_list = []
    for cat in rm_duplicate(data.topic_info['Category'].tolist()): 
        q = ','.join(str(x) for x in data.topic_info[data.topic_info['Category']==cat]["Term"].tolist())
        if cat != "Default":
            query_params = {'input': q}
            ollama_inference_count += 1
            topics_sentance = "Topic{}: ".format(ollama_inference_count)
            print("Topic{} seg_list:{}\n".format(ollama_inference_count, q))
            print("▶ ")
            
            # 根據不同的 LLM 類型處理回應
            if chain_type == "azure":
                for char in generate_response_for_query(chain, query_params):
                    if hasattr(char, 'content'):
                        print(char.content, end='', flush=True)
                        topics_sentance += char.content
                    else:
                        print(char, end='', flush=True)
                        topics_sentance += str(char)
            else:  # ollama
                response = chain.invoke(query_params)
                print(response, end='', flush=True)
                topics_sentance += response
                
            print("\n")
            topics_list.append(topics_sentance)
    return topics_list

def content_and_topic_relation_eda(data_frame, topic_list_sqg):
    data_frame["topics"] = None
    for Q in range(0, len(data_frame)):
        chunk_tags = data_frame["seg_list"][Q]
        keywords = rm_duplicate(chunk_tags) # terms_list is from seg_list
        # 統計每個topic中被hit到的次數
        hit_counts_per_topic = []
        
        for index, row in topic_list_sqg.iterrows():
            topic = row['Category']
            topic_list = row['topic_list_sqg']
            hit_count = sum(keyword in topic_list for keyword in keywords)
            hit_counts_per_topic.append({'Category': topic, 'HitCount': hit_count})
        
        # 創建結果的DataFrame
        result_df = pd.DataFrame(hit_counts_per_topic)
        sorted_result_df = result_df.sort_values(by='HitCount', ascending=False)
        # 計算HitCount的總數
        total_hit_count = sorted_result_df['HitCount'].sum()
        # 計算55%的總數
        percent = 0.55
        threshold = total_hit_count * percent
        # 依序加總，當數字達到總數的55%時，顯示Category有哪些
        cumulative_sum = 0
        categories_reaching_threshold = []
        for index, row in sorted_result_df.iterrows():
            cumulative_sum += row['HitCount']
            categories_reaching_threshold.append(row['Category'])
            if cumulative_sum >= threshold:
                break
        
        data_frame.at[Q, 'topics']  = categories_reaching_threshold
        if data_frame.at[Q, 'topics'] == []:
            data_frame.at[Q, 'topics'] = data_frame.at[i-1, 'topics']
    return data_frame

def elapsed_time(description, s_time):
    end_time = time.time()
    elapsed_time = round(end_time - s_time,2)
    return ("{} 花費: {} 秒".format(description, elapsed_time))

def Confirm_EmbeddingToken_is_Working(TIKTOKEN_CACHE_DIR, CACHE_KEY,embeddings):
    # Embedding 離線 token 準備
    # 離線載入 tiktoken : https://blog.csdn.net/qq_35054222/article/details/137127660
    # # blobpath = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    # # cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
    # # print(cache_key) # 9b5ad71b2ce5302211f9c61530b329a4922fc6a4
    os.environ["TIKTOKEN_CACHE_DIR"] = TIKTOKEN_CACHE_DIR
    print('TIKTOKEN 位置: {}'.format(os.getenv("TIKTOKEN_CACHE_DIR")))
    assert os.path.exists(os.path.join(TIKTOKEN_CACHE_DIR, CACHE_KEY))
    input_text = "Hello, world"
    encoding = tiktoken.get_encoding("cl100k_base")
    vector = embeddings.embed_query(input_text)
    return "輸入文字: {} ; Embedding 後: {}".format(input_text, vector[:3])

class Ckip:  # step 2: 文章斷字斷詞
    def __init__(self, STOP_WORDS_PATH):
    # 匯入停用詞
        with open(STOP_WORDS_PATH, "r", encoding="utf-8") as f:
            self.stopwords = [word.strip("\n") for word in f.readlines()]

        """
        離線下載: ( 先確認有: pip install huggingface_hub)
        1. huggingface-cli login
        2. huggingface-cli download --resume-download ckiplab/bert-base-chinese-ws --local-dir ./ckip_model/bert-base-chinese-ws
        3. huggingface-cli download --resume-download ckiplab/bert-base-chinese-pos --local-dir ./ckip_model/bert-base-chinese-pos
        4. huggingface-cli download --resume-download ckiplab/bert-base-chinese-ner --local-dir ./ckip_model/bert-base-chinese-ner
        """
        # # Initialize drivers(線上)
        # print("Initializing drivers ... WS")
        # ws_driver = CkipWordSegmenter(model="bert-base", device=0)
        # print("Initializing drivers ... POS")
        # pos_driver = CkipPosTagger(model="bert-base", device=0)
        # print("Initializing drivers ... NER")
        # ner_driver = CkipNerChunker(model="bert-base", device=0)
        # print("Initializing drivers ... all done")

        ## Initialize drivers (離線) 
        device = 0 if torch.cuda.is_available() else -1 # 應對 MacOS 不能使用 CUDA 的問題
        print("Initializing drivers ... WS")
        self.ws_driver = CkipWordSegmenter(model_name="./ckip_model/bert-base-chinese-ws", device=device)
        print("Initializing drivers ... POS")
        self.pos_driver = CkipPosTagger(model_name="./ckip_model/bert-base-chinese-pos", device=device)
        print("Initializing drivers ... NER")
        self.ner_driver = CkipNerChunker(model_name="./ckip_model/bert-base-chinese-ner", device=device)
        print("Initializing drivers ... all done")
        print()

    # 對文章進行斷詞
    def do_CKIP_WS(self, article):
        ws_results = self.ws_driver([str(article)])
        return ws_results
    
    # 對詞組進行詞性標示
    def do_CKIP_POS(self, ws_result):
        pos = self.pos_driver(ws_result[0])
        all_list = []
        for sent in pos:
            all_list.append(sent)
        return all_list
    
    # 保留名詞與動詞
    def _pos_filter(self, pos):
        for i in list(set(pos)):
            if i.startswith("N") or i.startswith("V"):
                return "Yes"
            else:
                continue
    
    # 去除數字與網址詞組
    def _remove_number_url(self, ws):
        number_pattern = "^\d+\.?\d*"
        url_pattern = "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$"
        space_pattern = "^ .*"
        num_regex = re.match(number_pattern, ws)
        url_regex = re.match(url_pattern, ws)
        space_regex = re.match(space_pattern, ws)
        if num_regex ==  None and url_regex == None and space_regex == None:
            return True
        else:
            return False
    
    # 執行資料清洗
    def cleaner(self, ws_results, pos_results):
        stopwords = self.stopwords
        word_lst = []
        for ws, pos in zip(ws_results[0], pos_results):
            in_stopwords_or_not = ws not in stopwords          #詞組是否存為停用詞
            if_len_greater_than_1 = len(ws) >= PHRASE_LENGTH   #詞組長度必須大於1
            is_V_or_N = self._pos_filter(pos)                        #詞組是否為名詞、動詞
            is_num_or_url = self._remove_number_url(ws)              #詞組是否為數字、網址、空白開頭
            if in_stopwords_or_not and if_len_greater_than_1 and is_V_or_N == "Yes" and is_num_or_url:
                word_lst.append(str(ws))
            else:
                pass
        return word_lst


class LDA_Category:
    def __init__(self, data_frame, TOPIC_SETTING, PASSES):
        self.seg_lst = data_frame["seg_list"].tolist()
        self.dictionary = corpora.Dictionary(self.seg_lst)
        self.corpus = [self.dictionary.doc2bow(i) for i in self.seg_lst]
        self.topic_setting = TOPIC_SETTING
        self.passes = PASSES
        
    def _coherence(self,i):
        self.ldamodel = LdaModel(corpus=self.corpus, num_topics=i, id2word=self.dictionary, passes=self.passes, random_state=42)
        ldacm = CoherenceModel(model=self.ldamodel, texts=self.seg_lst, dictionary=self.dictionary, coherence="c_v")
        return ldacm.get_coherence()
    
    def _coherence_count(self):
        self.x = range(1,self.topic_setting+1)
        self.y = [self._coherence(i) for i in self.x]
        return 

    def _gen_number_of_topics(self):
        top_five_indices = sorted(range(len(self.y)), key=lambda i: self.y[i], reverse=True)[:5]
        self.num_topics = sorted(top_five_indices, reverse= True)[0]
        print(self.num_topics)
        return 

    def gen_data_topic_info(self, SAVE_NAME, IMG_PATH):
        _ = self._coherence_count()
        _ = self._gen_number_of_topics()
        lda = LdaModel(self.corpus, num_topics = self.num_topics, id2word = self.dictionary, passes = self.passes ,random_state = 42) # passes = 30
        # pyLDAvis.enable_notebook()
        self.data = pyLDAvis_gensim_models.prepare(lda , self.corpus, self.dictionary)
        _ = self._save_ldavis(SAVE_NAME, IMG_PATH)
        _ = self._save_plot(SAVE_NAME, IMG_PATH)
        return self.data

    def _save_plot(self, SAVE_NAME, IMG_PATH):
        plt.plot(self.x,self.y)
        plt.xlabel("主題數目")
        plt.ylabel("coherence大小")
        plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        plt.title("主題-coherence變化情形")
        plt.show()
        plt.savefig("{}/twlife_{}分類_coherence變化情形.png".format(IMG_PATH,SAVE_NAME) )
        return 
        
    def _save_ldavis(self, SAVE_NAME, IMG_PATH):
        print(self.data)
        _ = pyLDAvis.save_html(self.data, "{}/twlife_{}分類.html".format(IMG_PATH,SAVE_NAME) )
        # 遇到問題，需修改套件程式碼: https://github.com/bmabey/pyLDAvis/issues/69 
        return 



""" 以下為 Create Ndoe Function"""

def create_Node_Product(kg, data_frame):
    node_count = 0
    for product in rm_duplicate(data_frame["filename"].tolist()):
        params = {'product': product}
        _ = kg.query(PRODUCT_QUERY, params=params)
        node_count += 1
    print(f"Created {node_count} nodes")
    return

def create_Node_Topics(kg, topics_list): # 建立 TOPICS數據庫  將列表拆解為 DataFrame
    topics_dict = {'topic': [], 'description': []}
    for item in topics_list:
        if '：' in item:
            topic, description = item.split('：', 1)
        elif ':' in item:
            topic, description = item.split(':', 1)
        else:
            topic, description = item.split(' ', 1)
        topics_dict['topic'].append(topic.strip())
        topics_dict['description'].append(description.strip())
    topics_datafrmae = pd.DataFrame(topics_dict)
    # topics_datafrmae.head()
    node_count = 0
    for index, row in topics_datafrmae.iterrows():
        params = {'topic': row['topic'], 'description': row['description']}
        _ = kg.query(TOPIC_QUERY, params=params)
        node_count += 1
    print(f"Created {node_count} nodes")
    return

def create_Node_Chunks(kg, data_frame):
    node_count = 0
    for index, row in data_frame.iterrows():
        params = {
                'content': row['content'], 
                'filename': row['filename'],
                'seg_list': row['seg_list'],
                'topics': row['topics'],
                'summary': row['summary'],
                }
        _ = kg.query(CHUNK_QUERY, params=params)
        node_count += 1
    print(f"Created {node_count} nodes")
    return

def create_Relation_Tpoic_Chunks(kg):
    _= kg.query(RELATION_QUERY_TC)
    return

def create_Relation_Product_Chunks(kg):
    _= kg.query(RELATION_QUERY_PC)
    return

def create_VecotrIndex_content(kg, embeddings):

    neo4j_vector_store = Neo4jVector.from_existing_graph(embedding=embeddings, 
                                    url=NEO4J_URI, 
                                    username=NEO4J_USERNAME, 
                                    password=NEO4J_PASSWORD, 
                                    database=NEO4J_DATABASE,
                                    index_name="emb_index",
                                    node_label='Chunk', 
                                    embedding_node_property='contentEmbedding', 
                                    text_node_properties=['content'])
    _ = kg.refresh_schema()
    print(kg.schema)
    _ = kg.query(BUILD_VECTOR_INDEX_CONTENT) 
    _ = kg.query(SHOWINDEX)
    return 

def create_Node_RuleTopics(kg,topics_list):
    # 建立 TOPICS數據庫: 
    # 將列表拆解為 DataFrame
    topics_dict = {'topic': [], 'description': []}
    for item in topics_list:
        if '：' in item:
            topic, description = item.split('：', 1)
        elif ':' in item:
            topic, description = item.split(':', 1)
        elif ' ：' in item:
            topic, description = item.split(' ：', 1)
        else:
            topic, description = item.split(' ', 1)
        topics_dict['topic'].append(topic.strip())
        topics_dict['description'].append(description.strip())
    topics_datafrmae = pd.DataFrame(topics_dict)
    # topics_datafrmae.head()
    node_count = 0
    for index, row in topics_datafrmae.iterrows():
        params = {'ruletopic': row['topic'], 'description': row['description']}
        _ = kg.query(RULETOPIC_QUERY, params=params)
        node_count += 1
    print(f"Created {node_count} nodes")
    return

def create_Node_PageTable(kg,data_frame):
    data_frame['summary'] = None
    for index , str_data in enumerate(data_frame['content'].tolist()):
        data = json.loads(str_data)
        formatted_string = dict_to_string(data)
        data_frame.at[index,'summary'] = formatted_string
    # data_frame.head()
    node_count = 0
    for index, row in data_frame.iterrows():
        params = {
                'content': row['content'], 
                'filename': row['filename'],
                'seg_list': row['seg_list'],
                'topics': row['topics'],
                'summary': row['summary'],
                'page': row['page'],
                }
        _ = kg.query(PAGETABLE_QUERY, params=params)
        node_count += 1
    print((f"Created {node_count} nodes"))
    return 

def create_Relation_RuleTpoic_Pagetable(kg):
    _ = kg.query(RELATION_QUERY_RTPT)
    return

def create_Relation_Product_Pagetable(kg):
    _ = kg.query(RELATION_QUERY_PPT)
    return 

def create_VecotrIndex_pagetable(kg, embeddings):
    neo4j_vector_store = Neo4jVector.from_existing_graph(embedding=embeddings, 
                                    url=NEO4J_URI, 
                                    username=NEO4J_USERNAME, 
                                    password=NEO4J_PASSWORD, 
                                    database=NEO4J_DATABASE,
                                    index_name="emb_index_rule",
                                    node_label='PageTable', 
                                    embedding_node_property='summaryEmbedding', 
                                    text_node_properties=['summary'])
    _ = kg.refresh_schema()
    print(kg.schema)
    _ = kg.query(BUILD_VECTOR_INDEX_PAGETABLE) 
    _ = kg.query(SHOWINDEX)
    return 
