
import logging

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as pyLDAvis_gensim_models
from langchain_core.prompts import ChatPromptTemplate


from ..dto_package.chunk import Chunk
from ..general_package.utility import check_memory
from ..llm_package.llm import llm_4o, llm_qwen, generate_response_for_query
from ..llm_package.prompt import TOPIC_SUMMARY_PROMPT
from ..ckip_package.ckip_module import ckip


LOGGER = logging.getLogger("TNPS")
MAX_TOPIC_NUMBER = 20
LDA_PASSES = 10
llm = llm_qwen if check_memory() else llm_4o
prompt_template = ChatPromptTemplate.from_messages(TOPIC_SUMMARY_PROMPT)
chain = prompt_template | llm


def lda_analysis(all_file_chunks: list[Chunk]):
    seg_lst = [chunk.segment_list for chunk in all_file_chunks]

    # seg_lst包含以下內容: seg_lst = [
    #     ["保險", "商品", "條款", "條款"],  # "條款" 出現兩次
    #     ["投保", "條款", "年齡", "限制"],
    #     ["保障", "範圍", "保險", "規定", "規定", "規定"]  # "規定" 出現三次
    # ]
    # 則 dictionary 可能會生成如下:
    # {'保險': 0, '商品': 1, '條款': 2, '投保': 3, '年齡': 4, '限制': 5, '保障': 6, '範圍': 7, '規定': 8}
    dictionary = corpora.Dictionary(seg_lst)
    # corpus的對應元素可能為 [
    #     [(0, 1), (1, 1), (2, 2)],           # "條款" 出現 2 次
    #     [(2, 1), (3, 1), (4, 1), (5, 1)],
    #     [(0, 1), (6, 1), (7, 1), (8, 3)]    # "規定" 出現 3 次
    # ]
    corpus = [dictionary.doc2bow(i) for i in seg_lst]

    def coherence(topic_number: int):
        LOGGER.info(f"Iterate through all corpus group numbers: {topic_number}")
        # 根據指定主題數量(理解上當成 corpus 分群數量)，建立LDA模型
        lda_model = LdaModel(corpus, num_topics=topic_number, id2word=dictionary, passes=LDA_PASSES)
        # 計算一致性模型
        ldacm = CoherenceModel(model=lda_model, texts=seg_lst, dictionary=dictionary, coherence="c_v")

        # 返回一致性值和LDA模型
        return ldacm.get_coherence(), lda_model

    # 計算不同主題數量(理解上當成 corpus 分群數量)時，對應的一致性數值
    topic_number_list = range(1, MAX_TOPIC_NUMBER + 1)  # 主題數量(理解上當成 corpus 分群數量)
    coherence_value_list = []   # 各主題數量對應的一致性值
    lda_model_list = []        # 各主題數量對應的LDA模型
    for i in topic_number_list:
        coherence_value, lda_model = coherence(i)

        lda_model_list.append(lda_model)
        coherence_value_list.append(coherence_value)
    LOGGER.info(f"topic_number->coherence_value: {dict(zip(topic_number_list, coherence_value_list))}")

    # 獲取一致性值最高的主題數量的索引
    best_num_of_topics_index = max(range(len(coherence_value_list)), key=lambda i: coherence_value_list[i])
    best_num_of_topics = topic_number_list[best_num_of_topics_index]
    LOGGER.info(f"best_num_of_topics: {best_num_of_topics}")

    # 建立最終LDA模型
    lda_model = lda_model_list[best_num_of_topics_index]

    # 準備LDA模型的可視化數據
    # pyLDAvis_gensim_models.prepare() 的 PreparedData 物件包含以下屬性：
    # 1. `topic_info` (DataFrame): 包含每個主題的資訊，包括主題的索引、詞彙和其權重。
    #     - `Term` (str): 詞彙的名稱。
    #     - `Category` (str): 該詞彙所屬的類別，"Default" 或 "Topic1/2/3"，區分通用詞與主題詞。
    #     - `Freq` (float): 該詞彙在主題中的頻率。
    #     - `Total` (float): 該詞彙在所有文檔中的總頻率。
    #     - `logprob` (float): 該詞彙的對數概率。
    #     - `loglift` (float): 該詞彙的提升度，表示該詞彙在主題中的重要性。
    # 2. `doc_info` (DataFrame): 包含每個文檔的資訊，包括文檔的索引和其對應的主題分佈。
    # 3. `vector` (ndarray): 包含每個文檔的詞彙向量表示，通常是稀疏矩陣格式。
    # 4. `mdsData` (DataFrame): 用於可視化的多維尺度分析結果，包含每個主題在二維空間中的坐標。
    # 5. `doc_lengths` (ndarray): 每個文檔的長度，即文檔中詞彙的數量。
    # 6. `term_frequency` (ndarray): 每個詞彙在所有文檔中的頻率。
    prepated_data = pyLDAvis_gensim_models.prepare(lda_model, corpus, dictionary)

    corpus_under_topic_df = prepated_data.topic_info[["Term", "Category"]]
    corpus_under_topic_df = corpus_under_topic_df[corpus_under_topic_df["Category"] != "Default"]

    corpus_under_topic_df["Category_int"] = corpus_under_topic_df["Category"].str.extract(r'(\d+)').astype(int)
    # 範例: [{'Term': '醫療', 'Category': 'Topic1', 'Category_int': 1}, ...]
    corpus_under_topic_df = corpus_under_topic_df.sort_values(by='Category_int')
    # 範例: {1: [胎兒', '疾病', '住院', '出院', '下列', '精神', ...], ...}
    corpus_under_topic_dict = corpus_under_topic_df.groupby('Category_int')['Term'].apply(list).to_dict()
    LOGGER.info(f"corpus_under_topic_dict: {corpus_under_topic_dict}")

    # 生成主題總結
    # 範例: {1: '商品條款的重點包括胎兒、疾病、住院、出院、精神...', ...}
    topic_summary_dict = {key: generate_response_for_query(chain, {"input": val}) for key, val in corpus_under_topic_dict.items()}
    LOGGER.info(f"topic_summary_dict: {topic_summary_dict}")

    # 將主題總結進行斷詞
    topic_summary_segment_dict = {key: ckip.process_flow(val) for key, val in topic_summary_dict.items()}
    LOGGER.info(f"topic_summary_segment_dict: {topic_summary_segment_dict}")

    # 分析每個 chunk 含有哪些 topic
    for chunk in all_file_chunks:
        segment_list = list(set(chunk.segment_list)) # 去除重複的詞彙

        hit_counts_per_topic = {}
        for idx, summary_segment_list in topic_summary_segment_dict.items():
            # 計算 segment_list 與 summary_segment_list 的交集
            intersection = set(segment_list) & set(summary_segment_list)
            # 計算交集的數量
            intersection_count = len(intersection)
            hit_counts_per_topic[idx] = intersection_count

        # 將 hit_counts_per_topic 轉換為 DataFrame
        hit_counts_per_topic_df = pd.DataFrame(list(hit_counts_per_topic.items()), columns=['Topic', 'HitCount'])
        # 根據 HitCount 進行排序
        hit_counts_per_topic_df.sort_values(by='HitCount', ascending=False, inplace=True)

        # 計算 HitCount 的總數
        total_hit_count = hit_counts_per_topic_df['HitCount'].sum()
        # 計算 55% 的總數
        percent = 0.55
        if total_hit_count > 0:
            threshold = total_hit_count * percent
        else:
            threshold = 1  # 至少確保選出一個主題

        # 依序加總，當數字達到總數的55%時，顯示Category有哪些
        cumulative_sum = 0
        categories_reaching_threshold = []
        for index, row in hit_counts_per_topic_df.iterrows():
            cumulative_sum += row['HitCount']  # 累加HitCount
            categories_reaching_threshold.append(f"Topic{row['Topic']}")  # 添加類別到列表
            # 當累加值達到閾值時停止
            if cumulative_sum >= threshold:
                break
        
        chunk.topic_list = categories_reaching_threshold
    
    return all_file_chunks, topic_summary_dict
        



