import os
import re
import json
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from ...dto_package.chunk import Chunk


LOGGER = logging.getLogger("TNPS")


def parse_policy_content(file_dir: str, file_name: str):
    """
    此函數用於處理政策文檔的分塊規則，主要處理包含【】標記的條文內容。
    處理流程：
    1. 合併所有頁面內容
    2. 按【】標記分割條文
    3. 處理前言部分
    4. 處理附表部分
    
    示例輸入：
    pages = [
        Document(page_content="前言內容【第一條】條文1內容【第二條】條文2內容附表1", metadata={"source": "xxx.pdf", "page": 1}),
        Document(page_content="附表2內容...", metadata={"source": "xxx.pdf", "page": 2})
    ]
    
    all_chunks = [
        "前言內容",
        "【第一條】條文1內容",
        "【第二條】條文2內容",
        "附表1",
        "附表2內容"
    ]

    示例輸出：
    [
        Chunk(content="前言內容", filename="xxx.pdf", page=[1]),
        Chunk(content="【第一條】條文1內容", filename="xxx.pdf", page=[1]),
        Chunk(content="【第二條】條文2內容", filename="xxx.pdf", page=[1]),
        Chunk(content="附表1", filename="xxx.pdf", page=[1]),
        Chunk(content="附表2內容", filename="xxx.pdf", page=[2])
    ]
    """
    # 讀取PDF文件
    loader = PyPDFLoader(os.path.join(file_dir, file_name))
    pages: list[Document] = loader.load_and_split()
    
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
    LOGGER.info(f"條文: {json.dumps(match, indent=2, ensure_ascii=False)}")
    
    # 補充第一個條文的內容
    # 例如：如果 all_content = "前言【條文1】內容1【條文2】內容2"
    # 那麼 first_chunk 將會是 "前言"
    pattern = r'^(.*?)【'
    first_chunk = re.findall(pattern, all_content, re.DOTALL)[0]
    LOGGER.info(f"first_chunk: {json.dumps(first_chunk, indent=2, ensure_ascii=False)}")
    
    # 將第一個條文插入到結果的最前面
    # 例如：match 現在是 ["【條文1】內容1", "【條文2】內容2"]
    # 插入後 match 將變成 ["前言", "【條文1】內容1", "【條文2】內容2"]
    match.insert(0, first_chunk)
    
    # 補充最後一筆資料的尾巴
    # 將最後一筆資料根據「附表」進行拆分
    # 例如：如果 match[-1] = "內容2附表1"
    # 那麼 split_entries 將會是 ["內容2", "1"]
    split_entries = match[-1].split('附表')
    LOGGER.info(f"split_entries: {json.dumps(split_entries, indent=2, ensure_ascii=False)}")
    
    # 將拆分後的資料依序補在結果的後面
    # 例如：match 現在是 ["前言", "【條文1】內容1", "【條文2】內容2"]
    # 插入後 match 將變成 ["前言", "【條文1】內容1", "【條文2】內容2", "內容2", "附表1"]
    all_chunks = match[:-1] + [split_entries[0]] + ['附表' + entry for entry in split_entries[1:]]
    LOGGER.info(f"all_chunks: {json.dumps(all_chunks, indent=2, ensure_ascii=False)}")
    
    ret_list: list[Chunk] = []
    # 用於追蹤當前處理的文本位置，確保按順序處理每個chunk
    current_pos = 0
    # 建立頁面內容和頁碼的映射，用於後續追蹤每個chunk所在的頁碼
    # 例如：{1: "第一頁內容", 2: "第二頁內容", ...}
    page_contents = {page.metadata["page"]: page.page_content for page in pages}
    
    # 說明取得 chunk page nums 的做法
    # 理論上 all_chunks 只是 all_content 分段的結果, 兩者的內容是相同的
    # 所以可以根據 chunk 開始跟結束位置 以及 每一頁的開始跟結束座標, 來判斷 chunk 有出現在哪些 page
    for chunk in all_chunks:
        # 取得本 chunk 起點在 all_content 中是第幾個 char
        # 從current_pos開始搜索，確保按順序處理每個chunk
        chunk_start = all_content.find(chunk, current_pos)
        # 取得本 chunk 終點在 all_content 中是第幾個 char
        chunk_end = chunk_start + len(chunk)
        # 更新current_pos為當前chunk的結束位置，用於下一個chunk的搜索
        current_pos = chunk_end
        
        # 用於存儲當前chunk跨越的所有頁碼
        chunk_pages = []
        # 用於追蹤在完整文本中的當前位置
        current_pos_in_text = 0
        
        # 遍歷每個頁面，確定chunk跨越了哪些頁面
        for page_num, page_content in page_contents.items():
            # 計算本頁面在 all_content 中起點跟終點是第幾個 char
            page_start = current_pos_in_text
            page_end = current_pos_in_text + len(page_content)
            
            # 檢查chunk是否與當前頁面有重疊
            # 如果chunk的起始位置小於頁面的結束位置，且chunk的結束位置大於頁面的起始位置
            # 則表示chunk跨越了這個頁面
            if (chunk_start < page_end and chunk_end > page_start):
                chunk_pages.append(page_num)
            
            # 更新current_pos_in_text為當前頁面的結束位置
            current_pos_in_text = page_end
        
        # 創建新的Chunk對象，包含內容、文件名和所有相關頁碼
        ret_list.append(Chunk(
            content=chunk,
            filename=file_name,
            page=chunk_pages,
            summary=summary_policy_content(chunk)
        ))
    
    return ret_list

def summary_policy_content(content: str):
    # 定義正則表達式，用於匹配條文標記
    pattern_articles = r"【(.*?)】"
    # 使用正則表達式查找所有匹配的條文
    summary_str = re.findall(pattern_articles, content)

    if summary_str:
        # 如果找到匹配的條文，去除標記並將結果存入 summary 欄位
        summary_str = summary_str[0].replace("【", "")
        summary_str = summary_str.replace("】", "")
        return summary_str
    else:
        # 如果未找到條文，檢查是否有附表內容
        pattern_appendix = r"^(附表.*?)(?=\n)"
        appendix_match = re.search(pattern_appendix, content)
        if appendix_match:
            # 如果找到附表內容，將其存入 summary 欄位
            appendix_str = appendix_match.group(1)
            return appendix_str
        else:
            # 如果都未找到，將原始內容存入 summary 欄位 (前言會把原本內容放入summary)
            return content 
    