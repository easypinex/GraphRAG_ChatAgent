import logging
from typing import List, Dict, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as pyLDAvis_gensim_models
from langchain_core.prompts import ChatPromptTemplate
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from ..dto_package.chunk import Chunk
from ..general_package.utility import check_memory
from ..llm_package.llm import llm_4o, llm_qwen, generate_response_for_query
from ..llm_package.prompt import TOPIC_SUMMARY_PROMPT
from ..ckip_package.ckip_module import ckip

LOGGER = logging.getLogger("TNPS")
MAX_TOPIC_NUMBER = 20
LDA_PASSES = 10
# 使用線程池處理LLM請求
MAX_WORKERS = 4
# 設置主題相似度閾值
TOPIC_SIMILARITY_THRESHOLD = 0.55

# 根據記憶體選擇LLM模型
llm = llm_qwen if check_memory() else llm_4o
prompt_template = ChatPromptTemplate.from_messages(TOPIC_SUMMARY_PROMPT)
chain = prompt_template | llm

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
        # 過濾掉只出現一次的詞
        dictionary.filter_extremes(no_below=2, no_above=0.5)
        # 創建語料庫
        corpus = [dictionary.doc2bow(i) for i in seg_lst]
        return dictionary, corpus
    except Exception as e:
        LOGGER.error(f"創建詞典和語料庫時發生錯誤: {str(e)}")
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
        LOGGER.error(f"計算一致性值時發生錯誤: {str(e)}")
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
            process_func = partial(generate_response_for_query, chain)
            # 並行處理所有主題
            topic_summary_dict = dict(
                executor.map(
                    lambda x: (x[0], process_func({"input": x[1]})),
                    corpus_under_topic_dict.items()
                )
            )
        return topic_summary_dict
    except Exception as e:
        LOGGER.error(f"處理主題總結時發生錯誤: {str(e)}")
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
        hit_counts_per_topic = {
            idx: len(set(segment_list) & set(summary_segment_list))
            for idx, summary_segment_list in topic_summary_segment_dict.items()
        }
        
        # 轉換為DataFrame並排序
        hit_counts_per_topic_df = pd.DataFrame(
            list(hit_counts_per_topic.items()),
            columns=['Topic', 'HitCount']
        ).sort_values(by='HitCount', ascending=False)
        
        # 計算閾值
        total_hit_count = hit_counts_per_topic_df['HitCount'].sum()
        threshold_value = max(total_hit_count * threshold, 1)
        
        # 累加計算達到閾值的主題
        cumulative_sum = 0
        categories_reaching_threshold = []
        for _, row in hit_counts_per_topic_df.iterrows():
            cumulative_sum += row['HitCount']
            categories_reaching_threshold.append(f"Topic{row['Topic']}")
            # 當累加的hit count達到閾值時，跳出迴圈
            if cumulative_sum >= threshold_value:
                break
        
        chunk.topic_list = categories_reaching_threshold
    except Exception as e:
        LOGGER.error(f"分析chunk主題時發生錯誤: {str(e)}")
        raise

def lda_analysis(all_file_chunks: List[Chunk]) -> Tuple[List[Chunk], Dict[int, str]]:
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
        coherence_results = []
        
        for i in topic_number_list:
            coherence_value, lda_model = calculate_coherence(i, corpus, dictionary, seg_lst)
            coherence_results.append((coherence_value, lda_model))
        
        # 獲取最佳主題數量
        best_num_of_topics_index = max(
            range(len(coherence_results)),
            key=lambda i: coherence_results[i][0]
        )
        best_num_of_topics = topic_number_list[best_num_of_topics_index]
        LOGGER.info(f"最佳主題數量: {best_num_of_topics}")
        
        # 使用最佳主題數量的LDA模型
        lda_model = coherence_results[best_num_of_topics_index][1]
        
        # 準備可視化數據
        prepated_data = pyLDAvis_gensim_models.prepare(lda_model, corpus, dictionary)
        
        # 處理主題信息
        corpus_under_topic_df = prepated_data.topic_info[["Term", "Category"]]
        corpus_under_topic_df = corpus_under_topic_df[corpus_under_topic_df["Category"] != "Default"]
        corpus_under_topic_df["Category_int"] = corpus_under_topic_df["Category"].str.extract(r'(\d+)').astype(int)
        corpus_under_topic_df = corpus_under_topic_df.sort_values(by='Category_int')
        
        # 創建主題詞字典
        corpus_under_topic_dict = corpus_under_topic_df.groupby('Category_int')['Term'].apply(list).to_dict()
        LOGGER.info(f"主題詞字典: {corpus_under_topic_dict}")
        
        # 生成主題總結
        topic_summary_dict = process_topic_summaries(corpus_under_topic_dict)
        LOGGER.info(f"主題總結字典: {topic_summary_dict}")
        
        # 對主題總結進行分詞
        topic_summary_segment_dict = {
            key: ckip.process_flow(val)
            for key, val in topic_summary_dict.items()
        }
        LOGGER.info(f"主題總結分詞字典: {topic_summary_segment_dict}")
        
        # 分析每個chunk的主題
        for chunk in all_file_chunks:
            analyze_chunk_topics(chunk, topic_summary_segment_dict)
        
        return all_file_chunks, topic_summary_dict
        
    except Exception as e:
        LOGGER.error(f"LDA分析過程中發生錯誤: {str(e)}")
        raise
