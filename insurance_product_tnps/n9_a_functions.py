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

def rm_duplicate(seq):
    """
    移除序列中的重複項目
    參數：
        seq: 包含可能重複項目的序列
    返回：
        list: 不包含重複項目的新列表
    """
    seen = set()  # 用於儲存已見過的項目
    seen_add = seen.add  # 優化添加項目的方法
    # 使用列表推導式來過濾重複項目
    # 對於序列中的每個項目 x，檢查它是否已經在 seen 集合中
    # 如果 x 不在 seen 中，則將其添加到 seen 中，並將 x 包含在返回的列表中
    # 否則，該項目將被忽略，從而實現去重
    return [x for x in seq if not (x in seen or seen_add(x))]  # 返回不重複的項目列表

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
    print(f"條文: {json.dumps(match, indent=2, ensure_ascii=False)}")
    
    # 補充第一個條文的內容
    # 例如：如果 all_content = "前言【條文1】內容1【條文2】內容2"
    # 那麼 first_chunk 將會是 "前言"
    pattern = r'^(.*?)【'
    first_chunk = re.findall(pattern, all_content, re.DOTALL)[0]
    print(f"first_chunk: {json.dumps(first_chunk, indent=2, ensure_ascii=False)}")
    
    # 將第一個條文插入到結果的最前面
    # 例如：match 現在是 ["【條文1】內容1", "【條文2】內容2"]
    # 插入後 match 將變成 ["前言", "【條文1】內容1", "【條文2】內容2"]
    match.insert(0, first_chunk)
    
    # 補充最後一筆資料的尾巴
    # 將最後一筆資料根據「附表」進行拆分
    # 例如：如果 match[-1] = "內容2附表1"
    # 那麼 split_entries 將會是 ["內容2", "1"]
    split_entries = match[-1].split('附表')
    print(f"split_entries: {json.dumps(split_entries, indent=2, ensure_ascii=False)}")
    
    # 將拆分後的資料依序補在結果的後面
    # 例如：match 現在是 ["前言", "【條文1】內容1", "【條文2】內容2"]
    # 插入後 match 將變成 ["前言", "【條文1】內容1", "【條文2】內容2", "內容2", "附表1"]
    match = match[:-1] + [split_entries[0]] + ['附表' + entry for entry in split_entries[1:]]
    print(f"final match: {json.dumps(match, indent=2, ensure_ascii=False)}")
    
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
    for r in chain.stream(query_params):
        yield r

def pickle_save(SAVE_PATH, data):
    with open(SAVE_PATH, 'wb') as file:
        pickle.dump(data, file)
    return 

def pickle_read(SAVE_PATH):
    with open(SAVE_PATH, 'rb') as file:
        data = pickle.load(file)
    return data

def list_pdf_from_file_path(DEFAULT_PATH):
    pdf_names = []
    for filename in os.listdir(DEFAULT_PATH):
        if filename.endswith('.pdf'):
            pdf_names.append(filename)
    return pdf_names

def pdf_to_pages(pdf_name):
    # 使用 pdfplumber 開啟指定的 PDF 檔案
    # pdf 物件的屬性包括：
    # - pages: 獲取 PDF 檔案中的所有頁面
    # - metadata: 獲取 PDF 檔案的元數據
    # - is_encrypted: 檢查 PDF 檔案是否被加密
    # - num_pages: 獲取 PDF 檔案的頁數
    pdf = pdfplumber.open(pdf_name) 
    # 獲取 PDF 檔案中的所有頁面
    # pages 物件的屬性包括：
    # - extract_text(): 提取頁面中的文本內容
    # - extract_tables(): 提取頁面中的表格資料
    #   例如：返回的資料格式為 [
    #                       [
    #                           ['表格1_row1_1', '表格1_row1_2'], 
    #                           ['表格1_row2_1', '表格1_row2_2']
    #                       ], 
    #                       [
    #                           ['表格2_row1_1', '表格2_row1_2'], 
    #                           ['表格2_row2_1', '表格2_row2_2']
    #                       ]
    #                      ]
    # - to_image(): 將頁面轉換為圖像
    # - page_number: 獲取當前頁面的頁碼
    pages = pdf.pages
    # 初始化一個字典，用來儲存每頁的表格資料
    table_with_page = dict()
    # 遍歷每一頁，並提取表格資料
    for index, page in enumerate(pages):
        # 從當前頁面提取表格資料
        nested_list = page.extract_tables()    # 取出文字
        # 如果提取到的表格資料不為空，則將其儲存到字典中
        if nested_list != []:
            table_with_page[index + 1] = nested_list
    # 返回包含每頁表格資料的字典
    return table_with_page

def data_split(table_with_page):
    """
    將PDF表格資料分割成每行一個DataFrame的格式
    
    參數:
        table_with_page: 包含每頁表格資料的字典
    
    返回:
        tuple: (dict_page_pd, page_list)
            - dict_page_pd: 包含每頁每行DataFrame的字典
            - page_list: 對應每行資料的頁碼列表
    
    示例:
        輸入 table_with_page = 例如: {
            1 (page_no): [
                [
                    ['表格1_row1_1', '表格1_row1_2'], 
                    ['表格1_row2_1', '表格1_row2_2']
                ], 
                [
                    ['表格2_row1_1', '表格2_row1_2', '表格2_row1_3'], 
                    ['表格2_row2_1', '表格2_row2_2', '表格2_row2_3']
                ]
            ],
            2 (page_no): [
                [
                    ['表格3_row1_1'], 
                    ['表格3_row2_1']
                ]
            ]
        }
        
        輸出:
        dict_page_pd = {
            1: [
                DataFrame([['表格1_標題1', '表格1_標題2'], ['表格1_內容1', '表格1_內容2']]),
                DataFrame([['表格2_標題1', '表格2_標題2', '表格2_標題3'], ['表格2_內容1', '表格2_內容2', '表格2_內容3']])
            ],
            2: [
                DataFrame([['表格3_標題1'], ['表格3_內容1']])
            ]
        }
        page_list = [1, 1, 2]
    """
    # 初始化一個字典，用來儲存每頁的資料框
    dict_page_pd = dict()
    # 初始化一個列表，用來儲存每頁的頁碼
    page_list = []
    
    # 遍歷每一頁及其對應的資料
    for page, data in table_with_page.items():
        # 初始化一個列表，用來儲存當前頁的資料框
        df_list = []
        # 將資料轉換為資料框
        df = pd.DataFrame(data[:])
        
        # 遍歷資料框的每一行
        for i in range(df.shape[0]):
            # 將每一行的資料框添加到df_list中
            df_list.append(df.iloc[[i]])
            # 將當前頁碼添加到page_list中
            page_list.append(page)
        
        # 將當前頁的資料框列表存入字典中
        dict_page_pd[page] = df_list
    
    # 返回包含每頁資料框的字典和頁碼列表
    return dict_page_pd, page_list

def pandas_to_dict_format(dict_page_pd): 
    """
    將包含DataFrame的字典轉換為包含JSON字符串的字典
    
    參數:
        dict_page_pd: 包含每頁DataFrame列表的字典
            例如: {
                1: [DataFrame1, DataFrame2],
            }
    
    返回:
        dict_page_str: 包含每頁JSON字符串列表的字典
            例如: {
                1: ["JSON字符串1", "JSON字符串2"],
                2: ["JSON字符串3"]
            }
    """
    # 初始化一個空字典，用於存儲結果
    dict_page_str = dict()
    
    # 遍歷輸入字典中的每一頁及其對應的DataFrame列表
    for page, dfs in dict_page_pd.items():
        # 初始化一個列表，用於存儲當前頁的JSON字符串
        str_list = []
        # 遍歷當前頁的每個DataFrame
        for _, df in enumerate(dfs):
            # 將DataFrame轉換為JSON字符串，確保中文字符正確顯示
            your_json = df.to_json(force_ascii=False)
            # 將JSON對象轉換為字符串格式
            stringformat = str(your_json)  # 如要轉回dict : json.loads(stringformat)
            # 將字符串添加到當前頁的列表中
            str_list.append(stringformat)
        # 將當前頁的JSON字符串列表存入結果字典
        dict_page_str[page] = str_list
    # 返回結果字典
    return dict_page_str

def load_pdf_to_dataframe(DEFAULT_PATH): # step 0: 先把資料轉pandas
    # 初始化一個空的列表，用來儲存所有內容
    ALL_CONTENTS = []
    # 初始化兩個空的列表，分別用來儲存檔案名稱和內容
    pdf_name_list = []
    content_str_list = []
    # 獲取指定路徑下所有PDF檔案的檔案名稱
    pdf_names = list_pdf_from_file_path(DEFAULT_PATH)
    
    # 遍歷每個檔案名稱
    for index, _ in enumerate(pdf_names):
        # 獲取當前檔案的完整路徑
        pdf_name = pdf_names[index]    

        # 使用PyPDFLoader載入PDF檔案並將其分割成頁面
        loader = PyPDFLoader(DEFAULT_PATH + pdf_name)
        pages = loader.load_and_split()

        # 對每一頁進行處理，提取出有用的內容
        ALL_CONTENTS = policy_product_chunk_rule(pages)

        # 將檔案名稱重複添加到dict_col1中，數量與ALL_CONTENTS相同
        pdf_name_list += [pdf_name] * len(ALL_CONTENTS)
        # 將提取的內容添加到content_str_list中
        content_str_list += ALL_CONTENTS

    # 將檔案名稱和內容組合成一個字典
    data_dict = {
        'filename': pdf_name_list,
        'content': content_str_list}
    
    # 檢查檔案名稱和內容的長度是否一致
    if len(pdf_name_list) == len(content_str_list):
        # 將 value 轉成 Series 之後整個 dict 轉成 DataFrame
        data_frame = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_dict.items()]))
        data_frame['summary'] = None
    else: 
        # 如果不一致，返回一個空的DataFrame並提示用戶檢查
        data_frame = pd.DataFrame()
        print("Dataframe is Empty, Please check ...")
        
    return data_frame

def load_rule_pdf_to_dataframe(DEFAULT_PATH2): 
    # 初始化一個空的列表，用來儲存所有內容
    ALL_CONTENTS = []
    # 初始化一個空的列表，用來儲存PDF檔案的名稱
    pdf_name_list = []
    # 初始化兩個空的列表，分別用來儲存內容和頁碼
    content_str_list = []
    page_no_list = []
    # 獲取指定路徑下所有PDF檔案的檔案名稱
    pdf_names = list_pdf_from_file_path(DEFAULT_PATH2)

    # 遍歷每個PDF檔案名稱
    for index, _ in enumerate(pdf_names):
        pdf_name = pdf_names[index]
        # 將PDF檔案轉換為頁面
        table_with_page_dict = pdf_to_pages(DEFAULT_PATH2 + "/" + pdf_name)
        print(f"table_with_page_dict")
        pprint(table_with_page_dict)
        # 將頁面數據分割為字典格式
        dict_page_df, page_list = data_split(table_with_page_dict)
        print(f"dict_page_df")
        pprint(dict_page_df)
        # 將字典格式的數據轉換為列表格式
        ALL_CONTENTS = pandas_to_dict_format(dict_page_df)
        print(f"ALL_CONTENTS")
        pprint(ALL_CONTENTS)

        # 初始化一個列表來儲存每頁的數據
        page_wtih_data_list = []
        # 遍歷每個頁面的數據
        for key, value in ALL_CONTENTS.items():
            page_wtih_data_list += (value)  # 將每頁的數據添加到列表中
        # 將檔案名稱重複添加到pdf_name_list中，數量與page_wtih_data_list相同
        pdf_name_list += [pdf_name] * len(page_wtih_data_list)
        # 將提取的內容添加到content_str_list中
        content_str_list += page_wtih_data_list
        # 將頁碼添加到page_no_list中
        page_no_list += page_list

    # 將檔案名稱、內容和頁碼組合成一個字典
    data_dict = {
        'filename': pdf_name_list,
        'content': content_str_list,
        'page': page_no_list
    }
    # 檢查檔案名稱和內容的長度是否一致
    if len(pdf_name_list) == len(content_str_list):
        # 將字典轉換為DataFrame，並添加額外的欄位
        data_frame = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_dict.items()]))
        data_frame["content_remodified"] = None  # 初始化content_remodified欄位為None
        data_frame['seg_list'] = None  # 初始化seg_list欄位為None
    else: 
        # 如果不一致，返回一個空的DataFrame並提示用戶檢查
        data_frame = pd.DataFrame()
        print("Dataframe is Empty, Please check ...")

    print(f"data_frame")
    pprint(data_frame)
    return data_frame  # 返回生成的DataFrame

def summary_manual_rule(data_frame): 
    # 遍歷資料框中的每一行，提取內容
    for index, str_data in enumerate(data_frame['content'].tolist()):
        # 定義正則表達式，用於匹配條文標記
        pattern_articles = r"【(.*?)】"
        # 使用正則表達式查找所有匹配的條文
        summary_str = re.findall(pattern_articles, str_data)
        if summary_str:
            # 如果找到匹配的條文，去除標記並將結果存入 summary 欄位
            summary_str = summary_str[0].replace("【", "")
            summary_str = summary_str.replace("】", "")
            data_frame.at[index, 'summary'] = summary_str
        else:
            # 如果未找到條文，檢查是否有附表內容
            pattern_appendix = r"^(附表.*?)(?=\n)"
            appendix_match = re.search(pattern_appendix, str_data)
            if appendix_match:
                # 如果找到附表內容，將其存入 summary 欄位
                appendix_str = appendix_match.group(1)
                data_frame.at[index, 'summary'] = appendix_str
            else:
                # 如果都未找到，將原始內容存入 summary 欄位 (前言會把原本內容放入summary)
                data_frame.at[index, 'summary'] = str_data 

    # 返回更新後的資料框
    return data_frame

def content_remodified(data_frame):
    """
    處理資料框中的內容，將JSON格式的內容轉換為字串格式
    
    參數：
        data_frame: 包含待處理內容的pandas DataFrame
        
    返回：
        data_frame: 處理後的DataFrame，新增了content_remodified欄位
    """
    # 初始化content_remodified欄位為None
    data_frame["content_remodified"] = None
    
    # 遍歷資料框中的每一行
    for index in data_frame.index:
        v_str_final = ""
        # 將JSON字串轉換為Python字典
        for _, value in (json.loads(data_frame["content"].iloc[index])).items():
            v_str_list = []
            # 遍歷字典中的每個值
            for _, v_list in value.items():
                if v_list is not None:
                    # 過濾掉None值
                    v_list = [x for x in v_list if x is not None]
                    # 將列表中的元素用逗號連接
                    v_str = ",".join(v_list)
                    v_str_list.append(v_str)
                    # 將所有字串再次用逗號連接
                    v_str = ",".join(v_str_list)
                else:
                    v_str = ""
            # 累加所有處理後的字串
            v_str_final += v_str
        # 將處理後的字串存入資料框
        data_frame["content_remodified"].iloc[index] = v_str_final
    
    return data_frame

def ckip_to_seglist(ckip, data_frame, content): 
    # 為資料框新增一個名為 'seg_list' 的欄位，初始值為 None
    data_frame['seg_list'] = None
    # 遍歷資料框中的每一行
    for i in range(len(data_frame)):
        # 使用 CKIP 進行斷詞，範例輸出: ['這', '是', '一', '個', '範例']
        ws_results = ckip.do_CKIP_WS(content.iloc[i])
        # 使用 CKIP 進行詞性標註，範例輸出: [('這', '代名詞'), ('是', '動詞'), ('一', '數詞'), ('個', '量詞'), ('範例', '名詞')]
        pos_results = ckip.do_CKIP_POS(ws_results)
        # 清理斷詞結果，保留重要詞性的詞彙，範例輸出: ['這', '是', '範例']
        word_lst = ckip.cleaner(ws_results, pos_results)
        # 將處理後的詞彙列表存入資料框的 'seg_list' 欄位
        data_frame.at[i, 'seg_list'] = word_lst
    # 返回更新後的資料框
    return data_frame 

def topic_lsit_segment(ckip, topics_summary_list, data_topic_info):
    """
    將主題列表進行斷詞處理並生成相應的資料框。
    
    參數：
        ckip: CKIP 斷詞工具的實例
        topics_summary_list: 包含主題的列表
        data_topic_info: 包含主題類別資訊的資料框
    
    返回：
        topic_list_sqg: 包含斷詞結果和類別的資料框
    """
    topics_list_seg_lst = []  # 用於儲存每個主題的斷詞結果
    for i in range(len(topics_summary_list)):
        # 使用 CKIP 進行斷詞
        ws_results = ckip.do_CKIP_WS(topics_summary_list[i])
        # 使用 CKIP 進行詞性標註
        pos_results = ckip.do_CKIP_POS(ws_results)
        # 清理斷詞結果，保留重要詞性的詞彙
        word_lst = ckip.cleaner(ws_results, pos_results)
        # 將清理後的詞彙列表添加到結果列表中
        topics_list_seg_lst.append(word_lst)
    
    # 將斷詞結果轉換為資料框並添加類別
    topic_list_sqg = pd.DataFrame({'topic_list_sqg': topics_list_seg_lst})
    topic_list_sqg["Category"] = rm_duplicate(data_topic_info["Category"].tolist())
    return topic_list_sqg  # 返回包含斷詞結果和類別的資料框

def Category_split(data):
    # 將輸入的字串根據 "Topic" 進行分割，並返回分割後的第二部分轉換為整數
    return int(data.split("Topic")[1])

def data_category_sort(data):
    """
    pyLDAvis_gensim_models.prepare() 的 PreparedData 物件包含以下屬性：

    1. `topic_info` (DataFrame): 包含每個主題的資訊，包括主題的索引、詞彙和其權重。
        - `Term` (str): 詞彙的名稱。
        - `Category` (str): 該詞彙所屬的類別，"Default" 或 "Topic1/2/3"，區分通用詞與主題詞。
        - `Freq` (float): 該詞彙在主題中的頻率。
        - `Total` (float): 該詞彙在所有文檔中的總頻率。
        - `logprob` (float): 該詞彙的對數概率。
        - `loglift` (float): 該詞彙的提升度，表示該詞彙在主題中的重要性。

    2. `doc_info` (DataFrame): 包含每個文檔的資訊，包括文檔的索引和其對應的主題分佈。
    3. `vector` (ndarray): 包含每個文檔的詞彙向量表示，通常是稀疏矩陣格式。
    4. `mdsData` (DataFrame): 用於可視化的多維尺度分析結果，包含每個主題在二維空間中的坐標。
    5. `doc_lengths` (ndarray): 每個文檔的長度，即文檔中詞彙的數量。
    6. `term_frequency` (ndarray): 每個詞彙在所有文檔中的頻率。
    """
    # 從資料中提取主題資訊，包括詞彙和類別
    data_topic_info = data.topic_info[["Term", "Category"]]
    # 過濾掉類別為 "Default" 的項目
    data_topic_info = data_topic_info[data_topic_info["Category"] != "Default"]
    # 將類別轉換為整數，以便於排序
    data_topic_info["Category_int"] = data_topic_info["Category"].apply(Category_split)
    # 根據整數類別進行排序
    data_topic_info = data_topic_info.sort_values(by='Category_int')
    # 返回排序後的主題資訊，範例: [{'Term': '醫療', 'Category': 'Topic1', 'Category_int': 1}, {'Term': '費用', 'Category': 'Topic2', 'Category_int': 2}]
    return data_topic_info

def topic_summary(chain, data):
    """
    根據不同的 LLM 類型生成主題總結
    
    Args:
        chain: LLM chain
        data: 主題資料
        chain_type: LLM 類型 ("azure" 或 "ollama")
    """
    ollama_inference_count = 0  # 計數器，用於計算 ollama 的推理次數
    topics_summary_list = []  # 用於儲存生成的主題總結列表
    for cat in rm_duplicate(data.topic_info['Category'].tolist()):  # 遍歷不重複的主題類別
        q = ','.join(str(x) for x in data.topic_info[data.topic_info['Category'] == cat]["Term"].tolist())  # 將該類別下的所有詞彙連接成一個字串
        if cat != "Default":  # 如果類別不是 "Default"
            query_params = {'input': q}  # 準備查詢參數
            ollama_inference_count += 1  # 增加推理計數
            topics_sentance = f"Topic{ollama_inference_count}: "  # 初始化主題句子
            print(f"Topic{ollama_inference_count} seg_list:{q}\n")  # 輸出當前主題的斷詞列表
            print("Topic Summary ▶ ")  # 輸出提示符
            
            for char in generate_response_for_query(chain, query_params):  # 生成回應
                if hasattr(char, 'content'):
                    # Azure 的回應
                    print(char.content, end='', flush=True)  # 輸出內容
                    topics_sentance += char.content  # 將內容添加到主題句子
                else:  # 如果沒有內容屬性
                    # Ollama 的回應
                    print(char, end='', flush=True)  # 輸出回應
                    topics_sentance += str(char)  # 將回應轉為字串並添加到主題句子
 
            print("\n")  # 輸出換行
            topics_summary_list.append(topics_sentance)  # 將生成的主題句子添加到主題總結列表
    return topics_summary_list  # 返回主題總結列表

def content_and_topic_relation_eda(data_frame, topic_list_sqg):
    # 初始化topics欄位為None
    data_frame["topics"] = None
    # 遍歷每一個資料行
    for Q in range(0, len(data_frame)):
        # 獲取當前行的斷詞標籤
        chunk_tags = data_frame["seg_list"][Q]
        # 去除重複的關鍵詞
        keywords = rm_duplicate(chunk_tags)  # terms_list is from seg_list
        # 統計每個topic中被hit到的次數
        hit_counts_per_topic = []
        
        # 遍歷每個主題類別
        for index, row in topic_list_sqg.iterrows():
            topic = row['Category']  # 獲取主題類別
            topic_list = row['topic_list_sqg']  # 獲取該主題的詞彙列表
            # 計算關鍵詞在主題詞彙列表中的出現次數
            hit_count = sum(keyword in topic_list for keyword in keywords)
            # 將結果添加到hit_counts_per_topic列表中
            hit_counts_per_topic.append({'Category': topic, 'HitCount': hit_count})
        
        # 創建結果的DataFrame
        result_df = pd.DataFrame(hit_counts_per_topic)
        # 根據HitCount進行排序
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
            cumulative_sum += row['HitCount']  # 累加HitCount
            categories_reaching_threshold.append(row['Category'])  # 添加類別到列表
            # 當累加值達到閾值時停止
            if cumulative_sum >= threshold:
                break
        
        # 將達到閾值的類別賦值給data_frame的topics欄位
        data_frame.at[Q, 'topics'] = categories_reaching_threshold
        # 如果topics為空，則使用前一行的topics
        if data_frame.at[Q, 'topics'] == []:
            data_frame.at[Q, 'topics'] = data_frame.at[i-1, 'topics']
    return data_frame  # 返回更新後的data_frame

def elapsed_time(description, s_time):
    end_time = time.time()
    elapsed_time = round(end_time - s_time,2)
    return f"{description} 花費: {elapsed_time} 秒"

def Confirm_EmbeddingToken_is_Working(TIKTOKEN_CACHE_DIR, CACHE_KEY,embeddings):
    # Embedding 離線 token 準備
    # 離線載入 tiktoken : https://blog.csdn.net/qq_35054222/article/details/137127660
    # # blobpath = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    # # cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
    # # print(cache_key) # 9b5ad71b2ce5302211f9c61530b329a4922fc6a4
    os.environ["TIKTOKEN_CACHE_DIR"] = TIKTOKEN_CACHE_DIR
    assert os.path.exists(os.path.join(TIKTOKEN_CACHE_DIR, CACHE_KEY))
    input_text = "Hello, world"
    encoding = tiktoken.get_encoding("cl100k_base")
    vector = embeddings.embed_query(input_text)
    return f"輸入文字: {input_text} ; Embedding 後: {vector[:3]}"

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
        # 初始化LDA_Category類別，接收資料框、主題設定和迭代次數
        self.seg_lst = data_frame["seg_list"].tolist()  # 將資料框中的斷詞列表轉換為列表
        # 建立字典，將斷詞列表轉換為一個字典對象
        # 字典的每個鍵是唯一的詞彙，值是該詞彙在語料庫中的ID
        # 例如，假設self.seg_lst包含以下內容: [['這', '是', '一', '個', '範例'], ['這', '是', '另一', '個', '範例']]
        # 則字典可能會生成如下:
        # {0: '這', 1: '是', 2: '一', 3: '個', 4: '範例', 5: '另一'}
        self.dictionary = corpora.Dictionary(self.seg_lst)
        # 將斷詞列表轉換為語料庫
        # self.corpus是由字典生成的語料庫，每個文檔都被轉換為一個詞彙ID和其出現次數的元組列表
        # 例如，假設self.seg_lst的第一個元素為['這', '是', '一', '個', '範例']，則self.corpus的對應元素可能為[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]
        self.corpus = [self.dictionary.doc2bow(i) for i in self.seg_lst]
        self.topic_setting = TOPIC_SETTING  # 設定主題數量
        self.passes = PASSES  # 設定LDA模型的迭代次數
        
    def _coherence(self, i):
        """
        LdaModel說明:

        corpus = [
            ["蘋果", "香蕉", "水果", "健康"],
            ["狗", "貓", "動物", "寵物"],
            ["電影", "電視", "娛樂"]
        ]

        # 指定 LdaModel 將詞彙分群成指定數量的主題，這邊分成 2 群
        lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

        最後，我們可以查看每個主題的關鍵詞，以及詞彙的權重：
        for idx, topic in lda_model.print_topics(-1):
            print(f"主題 {idx}: {topic}")

        例如，假設我們的模型識別出以下主題：
        [(0, '0.10*"蘋果" + 0.10*"香蕉" + 0.10*"水果" + 0.10*"健康"'),
        (1, '0.15*"狗" + 0.15*"貓" + 0.15*"動物" + 0.15*"寵物"'),
        (2, '0.20*"電影" + 0.20*"電視" + 0.20*"娛樂"')]
        """
        self.ldamodel = LdaModel(corpus=self.corpus, num_topics=i, id2word=self.dictionary, passes=self.passes, random_state=42)  # 建立LDA模型
        """
        # 假設我們發現主題 0 的一致性分數較高，而主題 1 的一致性分數較低
        # 這表明主題 0 的詞彙之間的關聯性較強，而主題 1 的詞彙之間的關聯性較弱。

        通常會選擇一致性高的主題數量
        """
        ldacm = CoherenceModel(model=self.ldamodel, texts=self.seg_lst, dictionary=self.dictionary, coherence="c_v")  # 計算一致性模型
        return ldacm.get_coherence()  # 返回一致性值
    
    def _coherence_count(self):
        # 計算不同主題數量下的一致性值
        self.topic_number = range(1, self.topic_setting + 1)  # 主題數量範圍
        self.coherence = [self._coherence(i) for i in self.topic_number]  # 計算每個主題數量的一致性值

    def _gen_number_of_topics(self):
        # 搜尋最高一致性的主題數量
        top_five_indices = sorted(range(len(self.coherence)), key=lambda i: self.coherence[i], reverse=True)[:5]  # 獲取前五高的一致性值的索引
        self.best_num_of_topics = sorted(top_five_indices, reverse=True)[0]  # 獲取最佳主題數量

    def gen_data_topic_info(self, SAVE_NAME, IMG_PATH):
        # 生成主題資料信息
        _ = self._coherence_count()  # 計算一致性值
        _ = self._gen_number_of_topics()  # 生成最佳主題數量
        lda = LdaModel(self.corpus, num_topics=self.best_num_of_topics, id2word=self.dictionary, passes=self.passes, random_state=42)  # 建立最終LDA模型
        # pyLDAvis_gensim_models.prepare用於準備LDA模型的可視化數據，輸出包含主題的分佈、詞彙的關聯性等信息，便於進行主題模型的分析和可視化。
        self.data = pyLDAvis_gensim_models.prepare(lda, self.corpus, self.dictionary)
        _ = self._save_ldavis(SAVE_NAME, IMG_PATH)  # 保存LDA可視化結果
        _ = self._save_plot(SAVE_NAME, IMG_PATH)  # 保存一致性圖
        
        return self.data  # 返回可視化數據

    def _save_plot(self, SAVE_NAME, IMG_PATH):
        # 保存一致性圖
        plt.plot(self.topic_number, self.coherence)  # 繪製一致性圖
        plt.xlabel("主題數目")  # X軸標籤
        plt.ylabel("coherence大小")  # Y軸標籤
        plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]  # 設定字體
        matplotlib.rcParams["axes.unicode_minus"] = False  # 顯示負號
        plt.title("主題-coherence變化情形")  # 圖表標題
        plt.show()  # 顯示圖表
        plt.savefig(f"{IMG_PATH}/twlife_{SAVE_NAME}分類_coherence變化情形.png")  # 保存圖表
        return 
        
    def _save_ldavis(self, SAVE_NAME, IMG_PATH):
        # 保存LDA可視化結果
        _ = pyLDAvis.save_html(self.data, f"{IMG_PATH}/twlife_{SAVE_NAME}分類.html")  # 保存為HTML文件
        # 遇到問題，需修改套件程式碼: https://github.com/bmabey/pyLDAvis/issues/69 
        return


""" 以下為 Create Ndoe Function"""

def create_Node_Product(kg, data_frame):
    """
    在Neo4j圖數據庫中創建產品節點
    
    參數:
        kg: Neo4jGraph對象，用於與Neo4j數據庫交互
        data_frame: 包含產品信息的DataFrame
    
    功能:
        1. 從data_frame中提取不重複的產品名稱(文件名)
        2. 為每個產品創建一個節點
        3. 打印創建的節點數量
    """
    node_count = 0  # 初始化節點計數器
    for product in rm_duplicate(data_frame["filename"].tolist()):  # 遍歷去重後的產品列表
        params = {'product': product}  # 設置查詢參數
        _ = kg.query(PRODUCT_QUERY, params=params)  # 執行Neo4j查詢創建產品節點
        node_count += 1  # 節點計數增加
    print(f"Created {node_count} nodes")  # 打印創建的節點數量
    return

def create_Node_Topics(kg, topics_list): 
    """
    在Neo4j圖數據庫中創建主題節點
    
    參數:
        kg: Neo4jGraph對象，用於與Neo4j數據庫交互
        topics_list: 包含主題信息的列表，格式為 "主題：描述" 或 "主題:描述" 或 "主題 描述"
    
    功能:
        1. 將主題列表解析為主題和描述
        2. 創建DataFrame存儲主題數據
        3. 為每個主題創建一個節點
        4. 打印創建的節點數量
    """
    # 初始化字典用於存儲主題和描述
    topics_dict = {'topic': [], 'description': []}
    
    # 遍歷主題列表，解析每個項目
    for item in topics_list:
        if '：' in item:  # 處理中文冒號分隔的情況
            topic, description = item.split('：', 1)
        elif ':' in item:  # 處理英文冒號分隔的情況
            topic, description = item.split(':', 1)
        else:  # 處理空格分隔的情況
            topic, description = item.split(' ', 1)
        
        # 將解析後的主題和描述添加到字典中
        topics_dict['topic'].append(topic.strip())
        topics_dict['description'].append(description.strip())
    
    # 將字典轉換為DataFrame
    topics_datafrmae = pd.DataFrame(topics_dict)
    # topics_datafrmae.head()  # 用於調試時查看數據
    
    # 初始化節點計數器
    node_count = 0
    
    # 遍歷DataFrame中的每一行，創建主題節點
    for index, row in topics_datafrmae.iterrows():
        # 設置查詢參數
        params = {'topic': row['topic'], 'description': row['description']}
        # 執行Neo4j查詢創建主題節點
        _ = kg.query(TOPIC_QUERY, params=params)
        # 節點計數增加
        node_count += 1
    
    # 打印創建的節點數量
    print(f"Created {node_count} nodes")
    return

def create_Node_Chunks(kg, data_frame):
    """
    在Neo4j圖數據庫中創建內容塊節點
    
    參數:
        kg: Neo4jGraph對象，用於與Neo4j數據庫交互
        data_frame: 包含內容塊數據的DataFrame
    
    功能:
        1. 遍歷DataFrame中的每一行數據
        2. 為每個內容塊創建一個節點
        3. 打印創建的節點數量
    """
    # 初始化節點計數器
    node_count = 0
    # 遍歷DataFrame中的每一行
    for index, row in data_frame.iterrows():
        # 設置查詢參數，包含內容塊的各種屬性
        params = {
                'content': row['content'],  # 內容文本
                'filename': row['filename'],  # 來源文件名
                'seg_list': row['seg_list'],  # 分詞列表
                'topics': row['topics'],  # 相關主題
                'summary': row['summary'],  # 內容摘要
                }
        # 執行Neo4j查詢創建內容塊節點
        _ = kg.query(CHUNK_QUERY, params=params)
        # 節點計數增加
        node_count += 1
    # 打印創建的節點數量
    print(f"Created {node_count} nodes")
    return

def create_Relation_Tpoic_Chunks(kg):
    """
    在Neo4j圖數據庫中創建主題與內容塊之間的關係
    
    參數:
        kg: Neo4jGraph對象，用於與Neo4j數據庫交互
    
    功能:
        執行預定義的查詢來建立主題和內容塊之間的關係
    """
    # 執行預定義的查詢，建立主題和內容塊之間的關係
    _= kg.query(RELATION_QUERY_TC)
    return

def create_Relation_Product_Chunks(kg):
    """
    在Neo4j圖數據庫中創建產品與內容塊之間的關係
    
    參數:
        kg: Neo4jGraph對象，用於與Neo4j數據庫交互
    
    功能:
        執行預定義的查詢來建立產品和內容塊之間的關係
    """
    # 執行預定義的查詢，建立產品和內容塊之間的關係
    _= kg.query(RELATION_QUERY_PC)
    return

def create_VecotrIndex_content(kg, embeddings):
    """
    為內容塊創建向量索引
    
    參數:
        kg: Neo4jGraph對象，用於與Neo4j數據庫交互
        embeddings: 嵌入模型，用於生成文本的向量表示
    
    功能:
        1. 從現有圖形創建Neo4j向量存儲
        2. 刷新圖數據庫模式
        3. 構建內容的向量索引
        4. 顯示索引信息
    """
    # 從現有圖形創建Neo4j向量存儲，設置相關參數
    neo4j_vector_store = Neo4jVector.from_existing_graph(
                                    embedding=embeddings,  # 嵌入模型
                                    url=NEO4J_URI,  # Neo4j數據庫URI
                                    username=NEO4J_USERNAME,  # 用戶名
                                    password=NEO4J_PASSWORD,  # 密碼
                                    database=NEO4J_DATABASE,  # 數據庫名稱
                                    index_name="emb_index",  # 索引名稱
                                    node_label='Chunk',  # 節點標籤
                                    embedding_node_property='contentEmbedding',  # 嵌入屬性名稱
                                    text_node_properties=['content'])  # 文本屬性名稱
    # 刷新圖數據庫模式
    _ = kg.refresh_schema()
    # 打印圖數據庫模式
    print(kg.schema)
    # 構建內容的向量索引
    _ = kg.query(BUILD_VECTOR_INDEX_CONTENT) 
    # 顯示索引信息
    _ = kg.query(SHOWINDEX)
    return 

def create_Node_RuleTopics(kg, topics_list):
    """
    在Neo4j圖數據庫中創建規則主題節點
    
    參數:
        kg: Neo4jGraph對象，用於與Neo4j數據庫交互
        topics_list: 包含規則主題信息的列表
    
    功能:
        1. 將主題列表解析為主題和描述
        2. 創建DataFrame存儲主題數據
        3. 為每個規則主題創建一個節點
        4. 打印創建的節點數量
    """
    # 初始化字典用於存儲主題和描述
    topics_dict = {'topic': [], 'description': []}
    # 遍歷主題列表，解析每個項目
    for item in topics_list:
        if '：' in item:  # 處理中文冒號分隔的情況
            topic, description = item.split('：', 1)
        elif ':' in item:  # 處理英文冒號分隔的情況
            topic, description = item.split(':', 1)
        elif ' ：' in item:  # 處理帶空格的中文冒號分隔的情況
            topic, description = item.split(' ：', 1)
        else:  # 處理空格分隔的情況
            topic, description = item.split(' ', 1)
        # 將解析後的主題和描述添加到字典中
        topics_dict['topic'].append(topic.strip())
        topics_dict['description'].append(description.strip())
    # 將字典轉換為DataFrame
    topics_datafrmae = pd.DataFrame(topics_dict)
    # 初始化節點計數器
    node_count = 0
    # 遍歷DataFrame中的每一行，創建規則主題節點
    for index, row in topics_datafrmae.iterrows():
        # 設置查詢參數
        params = {'ruletopic': row['topic'], 'description': row['description']}
        # 執行Neo4j查詢創建規則主題節點
        _ = kg.query(RULETOPIC_QUERY, params=params)
        # 節點計數增加
        node_count += 1
    # 打印創建的節點數量
    print(f"Created {node_count} nodes")
    return

def create_Node_PageTable(kg, data_frame):
    """
    在Neo4j圖數據庫中創建頁面表格節點
    
    參數:
        kg: Neo4jGraph對象，用於與Neo4j數據庫交互
        data_frame: 包含頁面表格數據的DataFrame
    
    功能:
        1. 為每行數據生成摘要
        2. 為每個頁面表格創建一個節點
        3. 打印創建的節點數量
    """
    # 初始化摘要列
    data_frame['summary'] = None
    # 遍歷DataFrame中的每一行，生成摘要
    for index, str_data in enumerate(data_frame['content'].tolist()):
        # 將JSON字符串轉換為字典
        data = json.loads(str_data)
        # 將字典轉換為格式化字符串
        formatted_string = dict_to_string(data)
        # 將格式化字符串設置為摘要
        data_frame.at[index, 'summary'] = formatted_string
    
    # 初始化節點計數器
    node_count = 0
    # 遍歷DataFrame中的每一行，創建頁面表格節點
    for index, row in data_frame.iterrows():
        # 設置查詢參數，包含頁面表格的各種屬性
        params = {
                'content': row['content'],  # 內容JSON
                'filename': row['filename'],  # 來源文件名
                'seg_list': row['seg_list'],  # 分詞列表
                'topics': row['topics'],  # 相關主題
                'summary': row['summary'],  # 內容摘要
                'page': row['page'],  # 頁碼
                }
        # 執行Neo4j查詢創建頁面表格節點
        _ = kg.query(PAGETABLE_QUERY, params=params)
        # 節點計數增加
        node_count += 1
    # 打印創建的節點數量
    print((f"Created {node_count} nodes"))
    return 

def create_Relation_RuleTpoic_Pagetable(kg):
    """
    在Neo4j圖數據庫中創建規則主題與頁面表格之間的關係
    
    參數:
        kg: Neo4jGraph對象，用於與Neo4j數據庫交互
    
    功能:
        執行預定義的查詢來建立規則主題和頁面表格之間的關係
    """
    # 執行預定義的查詢，建立規則主題和頁面表格之間的關係
    _ = kg.query(RELATION_QUERY_RTPT)
    return

def create_Relation_Product_Pagetable(kg):
    """
    在Neo4j圖數據庫中創建產品與頁面表格之間的關係
    
    參數:
        kg: Neo4jGraph對象，用於與Neo4j數據庫交互
    
    功能:
        執行預定義的查詢來建立產品和頁面表格之間的關係
    """
    # 執行預定義的查詢，建立產品和頁面表格之間的關係
    _ = kg.query(RELATION_QUERY_PPT)
    return 

def create_VecotrIndex_pagetable(kg, embeddings):
    """
    為頁面表格創建向量索引
    
    參數:
        kg: Neo4jGraph對象，用於與Neo4j數據庫交互
        embeddings: 嵌入模型，用於生成文本的向量表示
    
    功能:
        1. 從現有圖形創建Neo4j向量存儲
        2. 刷新圖數據庫模式
        3. 構建頁面表格的向量索引
        4. 顯示索引信息
    """
    # 從現有圖形創建Neo4j向量存儲，設置相關參數
    neo4j_vector_store = Neo4jVector.from_existing_graph(
                                    embedding=embeddings,  # 嵌入模型
                                    url=NEO4J_URI,  # Neo4j數據庫URI
                                    username=NEO4J_USERNAME,  # 用戶名
                                    password=NEO4J_PASSWORD,  # 密碼
                                    database=NEO4J_DATABASE,  # 數據庫名稱
                                    index_name="emb_index_rule",  # 索引名稱
                                    node_label='PageTable',  # 節點標籤
                                    embedding_node_property='summaryEmbedding',  # 嵌入屬性名稱
                                    text_node_properties=['summary'])  # 文本屬性名稱
    # 刷新圖數據庫模式
    _ = kg.refresh_schema()
    # 打印圖數據庫模式
    print(kg.schema)
    # 構建頁面表格的向量索引
    _ = kg.query(BUILD_VECTOR_INDEX_PAGETABLE) 
    # 顯示索引信息
    _ = kg.query(SHOWINDEX)
    return 
