import os
import sys
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    
from typing import Any, Dict, List, Tuple
import traceback

from langchain_core.documents import Document
import pandas as pd
import pdfplumber
from pdfplumber.page import CroppedPage, Page
from pdfplumber.table import Table
from langchain_community.document_loaders.base import BaseLoader

from logger.logger import get_logger
from prompts.prompts import TABLE_TO_NATURAL_LANGUAGE_PROMPT

logging = get_logger()

class PDFTablePyPlumberLoader(BaseLoader):
    def __init__(self, file_path, llm=None) -> None:
        self.file_path = file_path
        self.llm = llm

    def load(self, **open_kwargs) -> List[Document]:
        # 儲存待合併的表格
        merge_table_candidates: List[Table] = []
        # 紀錄每一頁的表格
        page_tables_list:List[List[Table]] = []
        # 儲存所有頁面的DocumentPacker
        document_packers:List[DocumentPacker] = []
        with pdfplumber.open(self.file_path, **open_kwargs) as pdf:
            pages = pdf.pages
            page_docpacker_dict:Dict[Page, DocumentPacker] = {}
            for page in pages:
                tables = page.find_tables(table_settings={})
                croped_text_page = page
                for table in tables:
                    croped_text_page = croped_text_page.outside_bbox(table.bbox)
                crop_bbox = self._get_text_crop_box(page, tables)
                croped_text_page = croped_text_page.crop(crop_bbox)
                text = croped_text_page.extract_text()
                filename = os.path.basename(self.file_path)
                document = Document(text, metadata={"source": filename, "page_number": page.page_number})
                page_tables_list.append(tables)
                document_packer = DocumentPacker(document, self.llm)
                document_packers.append(document_packer)
                page_docpacker_dict[page] = document_packer
                
                merge_table_candidates += self._get_merge_candidate(croped_text_page, tables)                
                
                for table in tables:
                    # text = table.extract() # for debug
                    # 如果該表格貼底部, 則略過
                    if table in merge_table_candidates:
                        continue
                    if len(merge_table_candidates) > 0 and table.bbox[1] - page.height / 15 < 0:
                        # 有待合併清單, 且當前表格上邊界接近上方
                        first_table = merge_table_candidates[0]
                        # 把所有待合併的表格併到第一個表個
                        for merge_table_candi in merge_table_candidates:
                            # 這裡注意DocumentPacker的tables預設是空陣列, 所以只有第一個表格會有資料
                            page_docpacker_dict[first_table.page].tables.append(merge_table_candi)
                        page_docpacker_dict[first_table.page].tables.append(table)
                        merge_table_candidates.clear()
                    else:
                        # 沒有貼底部, 且沒有待合併的表格
                        document_packer.tables.append(table)
                        
        if len(merge_table_candidates) > 0:
            # 還有剩餘的待合併清單
            first_table = merge_table_candidates[0]
            for merge_table_candi in merge_table_candidates:
                page_docpacker_dict[first_table.page].tables.append(merge_table_candi)
            merge_table_candidates.clear()
            
        result: List[Document] = []
        llm_tasks: List[Tuple[DocumentPacker, str]] = []
        for documnet_packer in document_packers:
            result.append(documnet_packer.table_parse())
            if self.llm:
                llm_tasks += documnet_packer.llm_tasks
        if self.llm:
            self._documnet_table_desc_with_llm(llm_tasks)
        return result
    
    def _get_merge_candidate(self, page: CroppedPage, tables: List[Table]) -> List[Table]:
        '''取得可能需要合併的表格候選人 通常代表表格接近底部'''
        candidates = []
        for table in tables:
            if (table.bbox[3]+ page.height / 15) > page.height:
                # 如果表格的下邊界接近底部
                candidates.append(table)
        return candidates

    def _documnet_table_desc_with_llm(self, llm_tasks: List[Tuple['DocumentPacker', str]]):
        '''將表格的描述加入到該頁結尾'''
        from pydantic import BaseModel, Field
        from langchain_core.prompts import ChatPromptTemplate
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        class KnowledgeEntity(BaseModel):
            description: str = Field(
                description="Markdown格式描述的完整資訊知識"
            )
        prompt = TABLE_TO_NATURAL_LANGUAGE_PROMPT
        desc_llm = self.llm.with_structured_output(
            KnowledgeEntity
        )
        desc_chain = prompt | desc_llm
        
        def table_description(document_packer: DocumentPacker, list_str: str):
            try:
                desc = desc_chain.invoke({"list_str": list_str}).description
                if desc:
                    document_packer.document.page_content += '\n\n' + desc
            except:
                logging.error(f'Process pdf_table read faild!, error:\n{e}')
                logging.error(traceback.format_exc())
                document_packer.document.page_content += '\n\n' + list_str
                

        futures = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submitting all tasks and creating a list of future objects
            for document_packer, list_str in llm_tasks:
                future = executor.submit(table_description, document_packer, list_str)
                futures.append(future)
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing documents"
            ):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f'Process pdf_table read faild!, error:\n{e}')
                    logging.error(traceback.format_exc())
                    
    
    def _get_text_crop_box(self, page, tables):
        '''
        將邊界及頁首頁尾去除, 取得剩餘的 bbox
        
        需考慮Table邊界, 若Table邊界在非常邊邊, 則不切Table為主
        '''
        crop_bbox = (page.width/20, page.height/20, page.width-page.width/20, page.height - page.height/20) # (x0, top, x1, bottom)
        # 避免 crop 到表格, 最大crop到表格寬高
        table_x0_min = 999999
        table_x1_max = 0
        table_y0_min = 999999
        table_y0_max = 0
        for table in tables:
            table_x0_min = min(table_x0_min, table.bbox[0])
            table_y0_min = min(table_y0_min, table.bbox[1])
            table_x1_max = max(table_x1_max, table.bbox[2])
            table_y0_min = max(table_x1_max, table.bbox[3])
        crop_bbox = (min(crop_bbox[0], table_x0_min), min(crop_bbox[1], table_y0_min), max(crop_bbox[2], table_x1_max), max(crop_bbox[3], table_y0_max))
        return crop_bbox
        
    def _get_merge_top_talbe(self, total_height, merge_table_candidate: List[Table]) -> Table | None:
        '''
        取得貼近頁面頂端的表格
        '''
        for idx, table in enumerate(merge_table_candidate):
            if table.bbox[1] - total_height / 15 < 0:
                return merge_table_candidate.pop(idx)
    
    
class DocumentPacker:
    '''
    給予頁面的所有表格, 轉換成文字
    如果有 LLM, 則會把表格文字加到 llm_tasks, 反之直接加到 page_content
    最終表格文字都會放在該頁面底部
    '''
    def __init__(self, document, llm):
        self.document: Document = document
        self.llm = llm
        self.tables: List[Table] = []
        self.tableareas: List[TableArea] = []
        self.llm_tasks: List[Tuple['DocumentPacker', str]] = []
        
    def table_parse(self) -> Document:
        self.calculate_sub_tables()
        self.llm_tasks.clear()
        for table in self.tableareas:
            df = pd.DataFrame(table.table.extract())
            concat_str = str(df.values.tolist())
            if self.llm:
                self.llm_tasks.append((self, concat_str))
            else:
                self.document.page_content += '\n\n' + concat_str
        return self.document
        
    def calculate_sub_tables(self):
        '''
        為了解決表格中還有表格的問題, 將小的Table歸屬於大Table
        
        因為暫時不處理表格中的表格(大表格還是會有小表格的文字, 只是失去小表格的內容), 
        
        這裏只是歸類
        '''
        tableareas: List[TableArea] = [TableArea(table) for table in self.tables]
        tableareas_sorted = sorted(tableareas, key=lambda table: table.area)
        # 遍歷所有的 tableareas，檢查每個表格是否位於另一個更大的表格中
        for i, table in enumerate(tableareas_sorted):
            for larger_table in tableareas_sorted[i+1:]:
                # 如果當前的表格嵌套在 larger_table 中，則將其歸屬於 larger_table 的子表格（sub_tables）
                if table.in_other_table(larger_table):
                    larger_table.sub_tables.append(table)
                    table.parent_table = larger_table
                    self.tables.remove(table.table)
                    break  # 一旦表格被歸屬，停止當前表格的進一步檢查
        self.tableareas = [table for table in tableareas]

class TableArea:
    def __init__(self, table: Table):
        self.table = table
        self.sub_tables: List[TableArea] = []
        self.parent_table: TableArea = None
    
    def in_other_table(self, other: 'TableArea'):
        if self.area >= other.area:
            return False
        if self.x1 >= other.x1 and self.x2 <= other.x2 and self.y1 >= other.y1 and self.y2 <= other.y2:
            return True
        return False
    
    @property
    def x1(self):
        return self.table.bbox[0]
    
    @property
    def y1(self):
        return self.table.bbox[1]
    
    @property
    def x2(self):
        return self.table.bbox[2]
    
    @property
    def y2(self):
        return self.table.bbox[3]
    
    @property
    def width(self):
        return self.x2 - self.x1
    
    @property
    def height(self):
        return self.y2 - self.y1
    
    @property
    def area(self):
        return self.width * self.height


if __name__ == '__main__':
    file_path = os.path.join('test', 'test_data', '台灣人壽龍實在住院醫療健康保險附約.pdf')
    llm = None
    # from langchain_openai import AzureChatOpenAI
    # llm = AzureChatOpenAI(
    #         azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    #         azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    #         openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    #         temperature=0
    #     )
    loader = PDFTablePyPlumberLoader(file_path, llm)
    pages = loader.load(pages=[7, 8, 9])
    for page in pages:
        print(page.page_content)
        print('-' * 40)