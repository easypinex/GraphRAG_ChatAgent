from n9_b_imports import *
from n9_b_configs import *

# 去重複
def rm_duplicate(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def extract_topics(text):
    # 正則表達式來找到所有 'topic' 後面跟著兩位數字的情況
    pattern =  r'topic\d{1,2}'
    return re.findall(pattern, text,  re.IGNORECASE)

# 定義一個分詞函數
def jieba_tokenizer(text):
    return jieba.lcut(text)

def tf_idf_simularity(documents, user_query):
    # 將文件和查詢轉換為 TF-IDF 向量
    vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer)
    tfidf_matrix = vectorizer.fit_transform(documents + [user_query])
    
    # 計算查詢與每個文件之間的餘弦相似度
    query_vector = tfidf_matrix[-1]  # 最後一個向量是查詢的向量
    similarities = cosine_similarity(query_vector, tfidf_matrix[:-1]).flatten()
    
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_score = similarities[sorted_indices]
    # sorted_indices
    # sorted_score
    # 找出 top相似的文件: 
    # most_similar_index = similarities.argmax()
    return sorted_indices, sorted_score


def ask_question(llm_stream, user_question, json_results_product, json_results_xxx, data_topic_info_xxx):

    ask_prompt0 = ChatPromptTemplate.from_messages(PROMPT_STEP1)
    
    
    ask_prompt1 = ChatPromptTemplate.from_messages(PROMPT_STEP2)
    
    ask_prompt2 = ChatPromptTemplate.from_messages(PROMPT_STEP3)
    
    while True:
        try:
            # 確認有無商品
            as_description0= ask_prompt0.format_messages(input=user_question, products =json_results_product )
            llm_recommand_product = llm_stream.invoke(as_description0).content
            llm_recommand_product = ast.literal_eval(llm_recommand_product)
            llm_recommand_product = [i for i in llm_recommand_product if i !=""]
            print_product = ("保險商品為: {}\n".format(llm_recommand_product))
            
            # 從問題確認主題:
            as_description1 = ask_prompt1.format_messages(input=user_question, topics =json_results_xxx )
            llm_recommand_topics = llm_stream.invoke(as_description1).content
            llm_recommand_topics = ast.literal_eval(llm_recommand_topics)
            
            if llm_recommand_topics ==["無"]:
                 print_method1 = ("Method1 ~ 提問問題無法找到對應主題\n")
            else:
                print_method1 = ("Method1 ~ 找到相關聯的主題為: {}\n".format(llm_recommand_topics))
            
            
            #從標籤搜尋確認主題
            as_description2 = ask_prompt2.format_messages(input=user_question )
            llm_keywords = llm_stream.invoke(as_description2).content #取得標籤
            if llm_keywords =="":
                llm_keywords = [user_question]
            else:
                llm_keywords = ast.literal_eval(llm_keywords) #轉 list
            
            # 已索引的文件
            grouped_terms  = data_topic_info_xxx.groupby('Category_int')['Term'].apply(lambda x: ','.join(map(str, x))).tolist()
            documents = grouped_terms 
            # 標籤查詢
            user_query = ",".join(llm_keywords)
            # 取得分數跟index
            sorted_indices, sorted_score = tf_idf_simularity(documents, user_query)
            
            llm_recommand_topics2 = []
            score_list= []
            for index in sorted_indices[:3]:
                llm_recommand_topics2.append("Topic"+ str(index+1))
            # print(top_list)
            for index in sorted_score[:3]:
                score_list.append(index)
            # print(score_list)
            print_method2 = ("Method2 ~ 標籤為:{} 。透過主題標籤找到相關聯的主題為: {}\n".format(llm_keywords, llm_recommand_topics2))
            
            
            llm_recommand_topics = rm_duplicate( llm_recommand_topics2 + llm_recommand_topics)
            llm_recommand_topics=[i for i in llm_recommand_topics if i!='無']
            # most_important_topics = list(set(llm_recommand_topics2) & set(llm_recommand_topics))
            print_llmtopics = ("LLM 搜尋結果主題:{}".format(llm_recommand_topics))
    
            break

        except Exception as e:
             llm_recommand_product= []
             llm_recommand_topics = []
             print("EXCEPTION .....")
             break
    # print(print_product)
    # print(print_method1)
    # print(print_method2)
    # print(print_llmtopics)

    return  llm_recommand_product, llm_recommand_topics



def get_query_result(kg, embeddings, query_info, user_question, llm_recommand_product, llm_recommand_topics, threshold):
    query_result = neo4j_vector_search(kg, embeddings, query_info, user_question, llm_recommand_product, llm_recommand_topics)
    query_result =[i for i in query_result if i['score'] >= threshold]
    return query_result


def neo4j_vector_search(kg, embeddings, query_info, question, llm_recommand_product, llm_recommand_topics):
    question_emb = embeddings.embed_query(question)
    cypher_query = query_info[0]
    index_name = query_info[1]
    node_label = query_info[2]

    similar = kg.query(cypher_query, 
                     params={
                      'product': llm_recommand_product,
                      'topics': llm_recommand_topics,  
                      'question': question_emb, 
                      'index_name': index_name, 
                      'node_label': node_label, 
                      'top_k': 1000})
    return similar


def combine_table(kg, query_result_by_pagetable):
    lsit_of_sets = []
    for query_result in query_result_by_pagetable:
        lsit_of_sets.append((query_result['filename'], query_result['page']))
    
    combine_table_data_dict = {}
    for key, value in lsit_of_sets:
        if key not in combine_table_data_dict:
            combine_table_data_dict[key] = []
        if value not in combine_table_data_dict[key]:
            combine_table_data_dict[key].append(value)
    
    # print(combine_table_data_dict)
    
    combine_table_data_list = [{'filename': key, 'pages': value} for key, value in combine_table_data_dict.items()]
    result = kg.query(COMBINE_QUERY, params= {'filename_page': combine_table_data_list})

    return result

def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(session_id, redis_url="redis://localhost:6379")

def AI_response(ID, chain_with_history, user_question, refrence):
    # Use the chain in a conversation
    response = chain_with_history.invoke(
        {"input": user_question, "refrence": refrence},
        config={"configurable": {"session_id": ID}},
    )
    return (response.content)



def ask_from_neo4j(redis_container_storage, llm_stream, kg, embeddings, user_question, query_info_by_xxx, json_results, data_topic_info_xxx, ResponseThredshold):
    product = []
    topics1 = []
    topics2 = []
    query_info_by_chunk = query_info_by_xxx[0]
    query_info_by_pagetable = query_info_by_xxx[1]
    query_info_by_chunk_Blanketsearch = query_info_by_xxx[2]

    json_results_product = json_results[0]
    json_results_topic = json_results[1]
    json_results_ruletopic = json_results[2]

    data_topic_info = data_topic_info_xxx[0]
    data_topic_info_by_rules = data_topic_info_xxx[1]

    # 依照 條款 topic node搜尋: 
    next_prodcut1, next_topics1 = ask_question(llm_stream, user_question, json_results_product, json_results_topic, data_topic_info)
    next_prodcut1 = rm_duplicate(next_prodcut1)
    next_topics1 = rm_duplicate(next_topics1)
    print("▶ 條款 topic node搜尋 next_topics1: {}".format(next_topics1))

    # 依照 投保規則 topic node 搜尋:  
    next_prodcut2, next_topics2 = ask_question(llm_stream, user_question, json_results_product, json_results_ruletopic, data_topic_info_by_rules)
    next_prodcut2 = rm_duplicate(next_prodcut2)
    next_topics2 = rm_duplicate(next_topics2)
    print("▶ 投保規則 topicrule node 搜尋 next_topics2: {}".format(next_topics2))

    # 合併:
    next_prodcut = next_prodcut1 + next_prodcut2
    next_prodcut = rm_duplicate(next_prodcut)

    if len(next_prodcut) > 0:
        for i, _ in enumerate(next_prodcut):
            redis_container_storage.rpush('product', next_prodcut[i])     
        last_n_elements = redis_container_storage.lrange('product', len(next_prodcut)-2*(len(next_prodcut)), -1)
        product = [element.decode('utf-8') for element in last_n_elements]

        # product = next_prodcut
        topics1 = next_topics1
        topics2 = next_topics2
        print("▶ next_prodcut 有值 : {} ".format(next_prodcut))

    else:
        next_prodcut = redis_container_storage.lindex('product', -1).decode('utf-8')
        print("▶ next_prodcut 無值 : {} ".format(product))

    print("▶ PRODUCT : {}  topics1 : {} ;  topics2 : {} ".format(next_prodcut, topics1, topics2))

    # (適用於模糊問題)
    query_result_by_chunk = get_query_result(kg, embeddings,
                                            query_info_by_chunk,
                                            user_question,
                                            next_prodcut,
                                            next_topics1,
                                            threshold=ResponseThredshold)
    
    query_result_by_pagetable = get_query_result(kg, embeddings,
                                            query_info_by_pagetable,
                                            user_question,
                                            next_prodcut,
                                            next_topics2,
                                            threshold=ResponseThredshold)
    # (適用於清楚問題)
    query_result_by_chunk2 = get_query_result(kg, embeddings,
                                            query_info_by_chunk_Blanketsearch,
                                            user_question,
                                            next_prodcut,
                                            next_topics1,
                                            threshold=0.75)

    
    query_result_by_chunk += query_result_by_chunk2

    # 將列表中的字典轉換為可哈希的形式
    def make_hashable(d):
        return frozenset((k, tuple(v) if isinstance(v, list) else v) for k, v in d.items())
    # 去除重複的字典
    query_result_by_chunk = [dict(t) for t in {make_hashable(d) for d in query_result_by_chunk}]

    print("▶ query_result_by_chunk 數量: {}".format(len(query_result_by_chunk)))
    print("▶ query_result_by_pagetable 數量: {}".format(len(query_result_by_pagetable)))

    concate_chunks_content = [ i['content']for i in query_result_by_chunk]
    concate_chunks_content = "\n".join(concate_chunks_content)

    concate_tables_content = [ i['content']for i in query_result_by_pagetable]
    concate_tables_content = "\n".join(concate_tables_content)

    if (len(query_result_by_chunk) > 0 ) or (len(query_result_by_pagetable)):
        print("\n▶ 任一種搜尋結果大於 0   \n")
        topics1 += next_topics1
        topics2 += next_topics2
    else:
        print("\n▶ 搜尋結果都為 0   \n")
        if len(product) > 0:
            topics1 = []
            topics2 = []
            
    topics1 = rm_duplicate(topics1)
    topics2 = rm_duplicate(topics2)
    
    return concate_chunks_content, concate_tables_content