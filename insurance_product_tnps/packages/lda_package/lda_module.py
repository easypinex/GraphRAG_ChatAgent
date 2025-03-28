import json
import logging
from typing import List, Dict, Tuple, Any

import pandas as pd
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim_models as pyLDAvis_gensim_models
from langchain_core.prompts import ChatPromptTemplate
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from ..dto_package.chunk import Chunk
from ..general_package.utility import check_memory
from ..llm_package.llm_module import LLM_4O, LLM_QWEN, generate_response_for_query
from ..llm_package.prompt import TOPIC_SUMMARY_PROMPT
from ..ckip_package.ckip_module import CKIP


LOGGER = logging.getLogger("TNPS")
MAX_TOPIC_NUMBER = 20
LDA_PASSES = 10
# 使用線程池處理LLM請求
MAX_WORKERS = 4
# 設置主題相似度閾值
TOPIC_SIMILARITY_THRESHOLD = 0.55

# 根據記憶體選擇LLM模型
LLM = LLM_QWEN if check_memory() else LLM_4O
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(TOPIC_SUMMARY_PROMPT)
CHAIN = PROMPT_TEMPLATE | LLM


def create_dictionary_and_corpus(seg_lst: List[List[str]]) -> Tuple[corpora.Dictionary, List[List[Tuple[int, int]]]]:
    """
    創建詞典和語料庫，並進行優化處理
    
    Args:
        seg_lst: 分詞後的文本列表
        範例: [
            ["保險", "商品", "條款", "條款"],  # "條款" 出現兩次
            ["投保", "條款", "年齡", "限制"],
            ["保障", "範圍", "保險", "規定", "規定", "規定"]  # "規定" 出現三次
        ]
        
    Returns:
        dictionary: 詞典對象
        範例: {'保險': 0, '商品': 1, '條款': 2, '投保': 3, '年齡': 4, '限制': 5, '保障': 6, '範圍': 7, '規定': 8}
        corpus: 語料庫列表 (wordid, word 出現次數)
        範例: [
            [(0, 1), (1, 1), (2, 2)],           # "條款" 出現 2 次
            [(2, 1), (3, 1), (4, 1), (5, 1)],
            [(0, 1), (6, 1), (7, 1), (8, 3)]    # "規定" 出現 3 次
        ]
    """
    try:
        # 創建詞典，過濾掉出現次數過少或過多的詞
        dictionary = corpora.Dictionary(seg_lst)
        # # 過濾掉只出現一次的詞
        # dictionary.filter_extremes(no_below=2, no_above=0.5)
        # 創建語料庫
        corpus = [dictionary.doc2bow(i) for i in seg_lst]
        return dictionary, corpus
    
    except Exception as e:
        LOGGER.error(f"create dictionary and corpus error: {str(e)}")
        raise

def calculate_coherence(
    topic_number: int,
    corpus: List[List[Tuple[int, int]]],
    dictionary: corpora.Dictionary,
    seg_lst: List[List[str]]
) -> Tuple[float, LdaModel]:
    """
    計算指定主題數量的一致性值和LDA模型
    
    Args:
        topic_number: 主題數量
        corpus: 語料庫
        dictionary: 詞典
        seg_lst: 分詞後的文本列表
        
    Returns:
        coherence_value: 一致性值
        lda_model: LDA模型
    """
    try:
        LOGGER.info(f"正在計算主題數量 {topic_number} 的一致性值")
        lda_model = LdaModel(
            corpus,
            num_topics=topic_number,
            id2word=dictionary,
            passes=LDA_PASSES,
            random_state=42  # 設置隨機種子以確保結果可重現
        )
        ldacm = CoherenceModel(
            model=lda_model,
            texts=seg_lst,
            dictionary=dictionary,
            coherence="c_v"
        )
        return ldacm.get_coherence(), lda_model
    except Exception as e:
        LOGGER.error(f"calculate coherence error: {str(e)}")
        raise

def process_topic_summaries(
    corpus_under_topic_dict: Dict[int, List[str]]
) -> Dict[int, str]:
    """
    使用線程池並行處理主題總結
    
    Args:
        corpus_under_topic_dict: 主題詞字典
        範例: {
            1: ["保險", "商品", "條款"],
            2: ["投保", "年齡", "限制"],
            3: ["保障", "範圍", "規定"]
        }
    Returns:
        topic_summary_dict: 主題總結字典
        範例: {
            1: "商品條款的重點包括保險費的繳納、保險單的承保責任、續保的條件與效力、通知與書面文件的送達方式、寬限期及契約的生效與終止等。",
            2: "投保年齡的限制，保障範圍的規定，以及保險費的計算方式。",
            3: "保障範圍的規定，以及保險費的計算方式。"
        }
    """
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 創建部分函數，固定chain參數
            process_func = partial(generate_response_for_query, CHAIN)
            # 並行處理所有主題
            topic_summary_dict = dict(
                executor.map(
                    lambda x: (x[0], process_func({"input": x[1]})),
                    corpus_under_topic_dict.items()
                )
            )
        return topic_summary_dict
    except Exception as e:
        LOGGER.error(f"process topic summaries error: {str(e)}")
        raise

def analyze_chunk_topics(
    chunk: Chunk,
    topic_summary_segment_dict: Dict[int, List[str]],
    threshold: float = TOPIC_SIMILARITY_THRESHOLD
) -> None:
    """
    分析單個chunk含有哪些主題
    
    Args:
        chunk: 文本塊對象
        topic_summary_segment_dict: 主題詞段字典
        threshold: 相似度閾值
    """
    try:
        # 去除重複詞
        segment_list = list(set(chunk.segment_list))
        
        # 計算每個主題的命中次數
        # hit_counts_per_topic: {1: 3, 2: 2, 3: 1}
        hit_counts_per_topic = {
            idx: len(set(segment_list) & set(summary_segment_list))
            for idx, summary_segment_list in topic_summary_segment_dict.items()
        }
        
        # 將字典轉換為列表並排序
        # sorted_topics: [(1, 3), (2, 2), (3, 1)]
        sorted_topics = sorted(
            hit_counts_per_topic.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 計算閾值
        # total_hit_count: 6
        total_hit_count = sum(count for _, count in sorted_topics)
        # threshold_value: 3
        threshold_value = max(total_hit_count * threshold, 1)
        
        # 累加計算達到閾值的主題
        cumulative_sum = 0
        topics_reaching_threshold = []
        for topic, count in sorted_topics:
            cumulative_sum += count
            topics_reaching_threshold.append(topic)
            # 當累加的hit count達到閾值時，跳出迴圈
            if cumulative_sum >= threshold_value:
                break
        
        chunk.topic_list = topics_reaching_threshold

    except Exception as e:
        LOGGER.error(f"analyze chunk topics error: {str(e)}")
        raise

def lda_analysis(all_file_chunks: List[Chunk], file_type: str) -> Tuple[List[Chunk], Dict[int, str]]:
    """
    執行LDA主題分析
    
    Args:
        all_file_chunks: 所有文本塊列表
        
    Returns:
        all_file_chunks: 更新後的文本塊列表
        topic_summary_dict: 主題總結字典
    """
    try:
        # 獲取分詞列表
        seg_lst = [chunk.segment_list for chunk in all_file_chunks]
        
        # 創建詞典和語料庫
        dictionary, corpus = create_dictionary_and_corpus(seg_lst)
        
        # 計算不同主題數量的一致性值
        topic_number_list = range(1, MAX_TOPIC_NUMBER + 1)
        coherence_results = [] # [(coherence_value, lda_model)]
        
        for i in topic_number_list:
            coherence_value, lda_model = calculate_coherence(i, corpus, dictionary, seg_lst)
            coherence_results.append((coherence_value, lda_model))
        # coherence_results 示例：
        # [(0.5, lda_model1), (0.6, lda_model2), (0.4, lda_model3), (0.7, lda_model4), (0.3, lda_model5), ...]
        # 其中每个元素是一个元组 (coherence_value, lda_model)
        
        coherence = [value for value, model in coherence_results]

        coherence_log = {i+1: value for i, value in enumerate(coherence)}
        LOGGER.info(f"[{file_type}] topic_num vs. coherence_value: {json.dumps(coherence_log, indent=2, ensure_ascii=False)}")

        # 获取前5个最佳一致性值的索引
        # 例如：如果 coherence_results 为 [(0.5, m1), (0.6, m2), (0.4, m3), (0.7, m4), (0.3, m5)]
        # 则 top_5_indices 可能为 [3, 1, 0, 2, 4]
        # 因为：
        # - coherence_results[3][0] = 0.7 (最大)
        # - coherence_results[1][0] = 0.6 (第二)
        # - coherence_results[0][0] = 0.5 (第三)
        # - coherence_results[2][0] = 0.4 (第四)
        # - coherence_results[4][0] = 0.3 (第五)
        top_5_indices = sorted(range(len(coherence_results)), key=lambda i: coherence_results[i][0], reverse=True)[:5]

        coherence_value_log = {index+1: coherence_results[index][0] for index in top_5_indices}
        LOGGER.info(f"[{file_type}] top 5 coherence value: {json.dumps(coherence_value_log, indent=2, ensure_ascii=False)}")

        # 在前5個中選擇主題數量最大的數值
        best_num_of_topics_index = sorted(top_5_indices, reverse=True)[0]  # 直接取前五個中索引值最大的
        best_num_of_topics = topic_number_list[best_num_of_topics_index]
        LOGGER.info(f"[{file_type}] best_num_of_topics: {best_num_of_topics}, coherence_value: {coherence_results[best_num_of_topics_index][0]}")
        
        # 使用最佳主題數量的LDA模型
        lda_model = coherence_results[best_num_of_topics_index][1]
        
        # 準備LDA模型的可視化數據
        #   pyLDAvis_gensim_models.prepare() 的 PreparedData 物件包含以下屬性：
        #     1. `topic_info` (DataFrame): 包含每個主題的資訊，包括主題的索引、詞彙和其權重。
        #         - `Term` (str): 詞彙的名稱。
        #         - `Category` (str): 該詞彙所屬的類別，"Default" 或 "Topic1/2/3"，區分通用詞與主題詞。
        #         - `Freq` (float): 該詞彙在主題中的頻率。
        #         - `Total` (float): 該詞彙在所有文檔中的總頻率。
        #         - `logprob` (float): 該詞彙的對數概率。
        #         - `loglift` (float): 該詞彙的提升度，表示該詞彙在主題中的重要性。
        #     2. `doc_info` (DataFrame): 包含每個文檔的資訊，包括文檔的索引和其對應的主題分佈。
        #     3. `vector` (ndarray): 包含每個文檔的詞彙向量表示，通常是稀疏矩陣格式。
        #     4. `mdsData` (DataFrame): 用於可視化的多維尺度分析結果，包含每個主題在二維空間中的坐標。
        #     5. `doc_lengths` (ndarray): 每個文檔的長度，即文檔中詞彙的數量。
        #     6. `term_frequency` (ndarray): 每個詞彙在所有文檔中的頻率。
        prepated_data = pyLDAvis_gensim_models.prepare(lda_model, corpus, dictionary)

        # 處理主題信息
        corpus_under_topic_df = prepated_data.topic_info[["Term", "Category"]]
        corpus_under_topic_df = corpus_under_topic_df[corpus_under_topic_df["Category"] != "Default"]
        corpus_under_topic_df["Category_num_str"] = corpus_under_topic_df["Category"].str.extract(r'(\d+)')
        corpus_under_topic_df = corpus_under_topic_df.sort_values(by='Category_num_str')
        
        # 創建主題詞字典
        # {
        #     "1": ["主題詞1", "主題詞2", "主題詞3"],
        #     "2": ["主題詞4", "主題詞5", "主題詞6"],
        #     "3": ["主題詞7", "主題詞8", "主題詞9"]
        # }
        corpus_under_topic_dict = corpus_under_topic_df.groupby('Category_num_str')['Term'].apply(list).to_dict()
        LOGGER.info(f"[{file_type}] corpus_under_topic_dict: {json.dumps(corpus_under_topic_dict, indent=2, ensure_ascii=False)}")
        
        # 生成主題總結
        # {
        #     "1": "主題詞1、主題詞2、主題詞3的重點包括...",
        #     "2": "主題詞4、主題詞5、主題詞6的重點包括...",
        #     "3": "主題詞7、主題詞8、主題詞9的重點包括..."
        # }
        topic_summary_dict = process_topic_summaries(corpus_under_topic_dict)
        LOGGER.info(f"[{file_type}] topic_summary_dict: {json.dumps(topic_summary_dict, indent=2, ensure_ascii=False)}")
        
        # 對主題總結進行分詞
        # {
        #     "1": ["主題詞1", "主題詞2", "主題詞3", "重點", "包括", "..."],
        #     "2": ["主題詞4", "主題詞5", "主題詞6", "重點", "包括", "..."],
        #     "3": ["主題詞7", "主題詞8", "主題詞9", "重點", "包括", "..."]
        # }
        topic_summary_segment_dict = {
            key: CKIP.process_flow(val)
            for key, val in topic_summary_dict.items()
        }
        LOGGER.info(f"[{file_type}] topic_summary_segment_dict: {json.dumps(topic_summary_segment_dict, indent=2, ensure_ascii=False)}")
        
        # 分析每個chunk所包含的主題
        # [
        #     Chunk(
        #         ...,
        #         topic_list=["1", "2", "3", ...]
        #     ),
        #     ...
        # ]
        for chunk in all_file_chunks:
            analyze_chunk_topics(chunk, topic_summary_segment_dict)
        
        return all_file_chunks, topic_summary_dict
        
    except Exception as e:
        LOGGER.error(f"[{file_type}] LDA analysis error: {str(e)}")
        raise
