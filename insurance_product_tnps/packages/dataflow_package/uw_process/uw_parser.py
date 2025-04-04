
import os
import json
import logging

import pdfplumber

from ...dto_package.chunk import Chunk


LOGGER = logging.getLogger("TNPS")


def parse_uw_content(file_dir: str, file_name: str):
    # 使用 pdfplumber 開啟指定的 PDF 檔案
    # pdf 物件的屬性包括：
    # - pages: 獲取 PDF 檔案中的所有頁面
    # - metadata: 獲取 PDF 檔案的元數據
    # - is_encrypted: 檢查 PDF 檔案是否被加密
    # - num_pages: 獲取 PDF 檔案的頁數
    pdf = pdfplumber.open(os.path.join(file_dir, file_name))

    # 獲取 PDF 檔案中的所有頁面
    # page 物件的屬性包括：
    # - extract_text(): 提取頁面中的文本內容
    # - extract_tables(): 提取頁面中的表格資料
    # - page_number: 獲取當前頁面的頁碼
    pages = pdf.pages
    ret_list: list[Chunk] = []
    for page in pages:
        # tables資料格式為 [
        #                   [
        #                       ['表格1_row1_1', '表格1_row1_2'], 
        #                       ['表格1_row2_1', '表格1_row2_2']
        #                   ], 
        #                   [   
        #                       ['表格2_row1_1', '表格2_row1_2'], 
        #                       ['表格2_row2_1', '表格2_row2_2']
        #                   ]
        #                ]
        tables = page.extract_tables()
        
        # table 的格式為：
        # [
        #     ['表格1_row1_1', '表格1_row1_2'], 
        #     ['表格1_row2_1', '表格1_row2_2']
        # ]
        for table in tables:        
            table_content = json.dumps(table, ensure_ascii=False)
            table_summary = summary_uw_content(table)
            ret_list.append(Chunk(
                content=table_content,
                filename=file_name,
                summary=table_summary,
                page=page.page_number,
            ))

    return ret_list

def summary_uw_content(table_rows: str):
    """
    table_rows 的格式為：
    [
        ["row1_1", "row1_2"], 
        ["row2_1", "row2_2"],
        ...
    ]

    輸出:
    '''
    row1_1: 
    row1_2

    row2_1:
    row2_2
    
    ...
    '''
    """
    # 單純將表格的content串成一個str
    ret_str = ""
    for row in table_rows:
        filter_row = [cell for cell in row if (cell is not None and cell != "")]
        if len(filter_row) > 0:
            ret_str += f"{filter_row[0]}:\n"
        if len(filter_row) > 1:
            ret_str += "\n".join(filter_row[1:])
        ret_str += "\n\n"
    return ret_str
