"""
============================================================
                        [ 參數 ]
============================================================
"""

RESPONSETHREDHOLD = 0.65

AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_API_VERSION = "2024-10-01-preview"
AZURE_OPENAI_ENDPOINT = "https://sales-chatbot-llm.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "GPT4o" 

AZURE_EMB_MODLE_API_KEY = ""
AZURE_EMB_API_VERSION = "2024-02-01"
AZURE_EMB_ENDPOINT = "https://sales-chatbot-llm.openai.azure.com/"
AZURE_EMB_MODLE = 'text-embedding-3-small'
AZURE_EMB_DEPLOYMENT = 'text-embedding-3-small'

NEO4J_URI = "neo4j://localhost:7688"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "neo4j"

"""
============================================================
                       [ 腳本 ]
============================================================
"""
QUERY_GET_FILENAME = """
MATCH (p:Product)
RETURN p.product AS product
"""

QUERY_GET_TOPICS = """
MATCH (t:Topic)
RETURN t.topic AS topic, t.description AS description
"""

QUERY_GET_RULETOPICS = """
MATCH (t:RuleTopic)
RETURN t.ruletopic AS ruletopic, t.description AS description
"""

VECTOR_SEARCH_BY_CHUNK = """
        WITH $topics AS topics, $product AS product
        CALL db.index.vector.queryNodes($index_name, $top_k, $question) YIELD node, score
        WHERE ANY(topic IN node.topics WHERE topic IN topics)
        WITH node, score, product
        CALL apoc.do.case(
          [
            size(product) > 0, 
            'UNWIND product AS prod
             MATCH (node:Chunk)-[:FROM]->(p:Product {product: prod})
             RETURN score,
                    node,
                    p.product AS product'
          ],
          'RETURN score,
                  node,
                  NULL AS product',
          {node: node, score: score, product: product}) YIELD value
        
        RETURN value.score AS score, 
               value.node.filename AS filename, 
               value.node.content AS content, 
               value.node.summary AS summary, 
               value.node.topics AS topics, 
               value.product AS product
        """

VECTOR_SEARCH_BY_PAGETABLE = """
        WITH $topics AS topics, $product AS product
        CALL db.index.vector.queryNodes($index_name, $top_k, $question) YIELD node, score
        WHERE ANY(rulet IN node.ruletopics WHERE rulet IN topics)
        WITH node, score, product
        CALL apoc.do.case(
          [
            size(product) > 0, 
            'UNWIND product AS prod
             MATCH (node:PageTable)-[:FROM]->(p:Product {product: prod})
             RETURN score,
                    node,
                    p.product AS product'
          ],
          'RETURN score,
                  node,
                  NULL AS product',
          {node: node, score: score, product: product}) YIELD value
        
        RETURN value.score AS score, 
               value.node.filename AS filename, 
               value.node.content AS content, 
               value.node.summary AS summary, 
               value.node.ruletopics AS ruletopics, 
               value.node.page AS page, 
               value.product AS product
        """

VECTOR_SEARCH_BY_CHUNK_BLANKETSEARCH = """
        WITH $product AS product
        UNWIND product AS prod
        CALL db.index.vector.queryNodes($index_name, $top_k, $question) YIELD node, score
        WITH node, score, prod
        MATCH (node:Chunk)-[:FROM]->(p:Product {product: prod})
        RETURN score, 
               node.filename AS filename, 
               node.content AS content, 
               node.summary AS summary, 
               node.topics AS topics, 
               p.product AS product
        """

COMBINE_QUERY = """
    WITH $filename_page AS data
    UNWIND data AS item
    UNWIND item.pages AS page
    MATCH (n:PageTable {filename: item.filename, page: page})
    RETURN n.filename AS filename, 
           n.content AS content, 
           n.summary AS summary, 
           n.ruletopics AS ruletopics, 
           n.page AS page
    """

LLM_RAG_PROMPT = [
        ("system", "你的檢索資料有: {refrence} "),
        ("system", "你的回答先參考檢索資料再回答，不要亂產生無關答案"),
        ("system", "如果問題跟檢索資料 {refrence} 無關，完全依照提供的資料回答問題, 不要亂產生答案"),
        ("human", "{input}")]


PROMPT_STEP1 =[
        ('system', '你是保險商品經紀人'),
        ('system', '先確認user input裡是否有提及到products內的資訊'),
        ('system', """提供以下幾個範例: 
                  input: '我需要提供哪些文件或資訊來完成保險金的申請'
                  回應:['']
    
                  input: '金窩心100裡長短期照顧的比較'
                  回應:['(排版)橫式條款-台灣人壽金窩心100長期照顧終身健康保險']
    
                  input: '我想詢問商品臻美滿美元有關付約撤銷的方法'
                  回應:['(排版)橫式條款-台灣人壽臻美滿美元利率變動型終身壽險.pdf']
         
                  input: '美世多美投保年齡'
                  回應:['(排版)橫式條款-台灣人壽美世多美元利率變動型終身壽險.pdf']
         
                  input: '鑫美元附約相關'
                  回應:['(排版)橫式條款-台灣人壽美年有鑫美元利率變動型還本終身保險.pdf']
                    """),
        ('user', 'products: \n\n {products}'),   
        ('user', 'input: \n\n {input}'),
    ]


PROMPT_STEP2 =[
        ('system', '你是保險商品經紀人'),
        ('system', '先理解user詢問的問題，是想知道壽險商品的哪個部分，並參考topics內容後，告知最合適的topics，如果沒合適的就回答無，不要有其他文字說明'),
        ('system', """提供以下幾個範例: 
                  input: '您好，我想了解一下如何撤銷我保單上的附加條款'
                  回應:['topic1, topic3']
    
                  input: '我考慮取消一些附加保障，請問這會如何影響我的保費'
                  回應:['topic2, topic9, topic16']
    
                  input: '請問我可以取消保單中的附加保險嗎？如果可以，流程是什麼？'
                  回應:['topic7']
    
                  input: '想知道開心的方式'
                  回應:['無']
                  
                    """),
        ('user', 'topics: \n\n {topics}'),   
        ('user', 'input: \n\n {input}'),
    ]


PROMPT_STEP3 =[
        ('system', '你是保險商品經紀人'),
        ('system', '依照input給我關鍵字就好，給個關鍵字都只能用逗號隔開'),
         ('system', """提供以下幾個範例: 
                  input: '您好，我想了解一下如何撤銷我保單上的附加條款'
                  回應:['撤銷,附加,條款']
    
                  input: '我考慮取消一些附加保障，請問這會如何影響我的保費'
                  回應:['附加,取消,保費']
    
                  input: '請問我可以取消保單中的附加保險嗎？如果可以，流程是什麼？'
                  回應:['附加,取消']
                    """),
    
        ('user', 'input: \n\n {input}'),
    ]
