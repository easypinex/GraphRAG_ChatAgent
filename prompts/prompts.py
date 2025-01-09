from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

### 讀取檔案, 請 LLM 以自然語言輸出表格內容
TABLE_TO_NATURAL_LANGUAGE_PROMPT = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是一個資訊整理專家，以Markdown格式輸出以下知識，但是不使用表格輸出，並包含完整資訊，以繁體中文回應",
            ),
            (
                "human",
                """請整理以下表格內容，並以文字輸出，並保留完整知識\n{list_str}""",
            ),
        ])


from langchain_experimental.graph_transformers.llm import system_prompt
### 給予Chunk提供 Entity與Relationship的描述
CHUNK_TO_ENTITY_AND_RELATIONSHIP_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "system",
                "請確保使用繁體中文回答問題"
            ),
            (
                "human",
                (
                    "Tip: Make sure to answer in the correct format and do "
                    "not include any explanations. "
                    "Use the given format to extract information from the "
                    "following input: {input}"
                ),
            ),
        ]
    )

### 尋找具有潛在重複 ID 的節點
PROMPT_FIND_DUPLICATE_NODES_SYSTEM = \
"""
You are a data processing assistant. Your task is to identify duplicate entities in a list and decide which of them should be merged.
The entities might be slightly different in format or content, but essentially refer to the same thing. Use your analytical skills to determine duplicates.

Here are the rules for identifying duplicates:
1. Entities with minor typographical differences should be considered duplicates, except when they refer to differences such as "new" vs. "old," or "initial" vs. "renewal." In these cases, do not merge the results.
2. Entities with different formats but the same content should be considered duplicates.
3. Entities that refer to the same real-world object or concept, even if described differently, should be considered duplicates.
4. If it refers to different numbers, dates, or products, do not merge results
"""
PROMPT_FIND_DUPLICATE_NODES_USER = \
"""
Here is the list of entities to process:
{entities}

Please identify duplicates, merge them, and provide the merged list.
"""

### 針對社群 - 關係 - 實體 - Chunk 等等關係進行摘要(Commnuity)
COMMNUITY_SUMMARY_SYSTEM = \
"""
請根據提供同社區的資訊包含 nodes 與 relationships, 產生同社區的自然語言的摘要資訊, 
請整理共通資訊與各別檔案的資訊，若僅有一個檔案則直接摘要即可. No pre-amble.
"""

## 整理歷史紀錄成為新問題
QUESTION_HISTORY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", (
            "Given a chat history and the latest user question " # 給定一段聊天歷史和使用者的最新問題
            "which might reference context in the chat history, " # 這個問題可能會引用聊天歷史中的上下文
            "formulate a standalone question which can be understood " # 請將問題重新表述為一個獨立的問題，使其在沒有聊天歷史的情況下也能被理解
            "without the chat history. Do NOT answer the question, " # 不要回答這個問題
            "just reformulate it if needed and otherwise return it as is." # 只需在必要時重新表述問題，否則原樣返回
            "請確保使用繁體中文回應"
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

## 找出對應的檔案ID
RELATED_FILE_IDS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", (
            "請根據提供的問題以及商品清單找出有關問題的所有商品清單ID(s)"
        )),
        ("human", "商品清單:{file_list}\n\n問題: {question}\n\n "),
    ]
)

### 針對詢問搜尋
QUESTION_PROMPT = ChatPromptTemplate.from_template(
"""
你是一個有用的助手, 你的任務是整理提供的資訊, 使用長度與格式符合「multiple paragraphs」針對使用者的問題來回應,
提供的資訊包含 檢索內容、圖譜資料庫相關節點與關係資訊, 無關的資訊直接忽略
你必須使用繁體中文回應問題, 盡可能在500字內回應,
若提供的資訊全部無關聯, 回應「找不到相關資訊」並結束, 不要捏造任何資訊,
最終的回應將清理後的訊息合併成一個全面的答案，針對回應的長度和格式對所有關鍵點和含義進行解釋
根據回應的長度和格式適當添加段落和評論。以Markdown格式撰寫回應。
回應應保留原有的意思和使用的情態動詞，例如「應該」、「可以」或「將」。
請確保使用繁體中文回答問題


以下為檢索內容:
"{context}"

以下為圖譜資料庫相關節點(Entities)、關係(Relationships)、社群(Reports)、Chunks(內文節錄)資訊:
"{graph_result}"

問題: {question}
"""
)

