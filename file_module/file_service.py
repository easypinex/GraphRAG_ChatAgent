
if __name__ == '__main__':
    import sys
    sys.path.append("..")
    
from typing import List

from langchain_core.documents import Document
from file_module.file_loaders.pdf_table_pyplumber_loader import PDFTablePyPlumberLoader
from database import db_session
from models.file_task import FileTask


'''
檔案操作的方法
'''
class FileService:
    def __init__(self, llm = None):
        self.llm = llm
        
    def read_file(self, file_path: str, pages: list[int] = None, load_kwargs = None) -> List[Document]:
        '''
        提供讀取檔案操作
        '''
        supports_exts = ['pdf']

        exts = file_path.split('.')
        if len(exts) == 0:
            raise ValueError(f"檔案名稱解析錯誤: {file_path}")
        
        ext = exts[-1].lower()
        if ext not in supports_exts:
            raise NotImplementedError(f"不支援的檔案類型: {file_path}, ext: {ext}")
        
        if ext == 'pdf':
            loader = PDFTablePyPlumberLoader(file_path, self.llm)

        if load_kwargs is None:
            load_kwargs = {}

        if pages is not None:
            load_kwargs['pages'] = pages
            
        pages = loader.load(**load_kwargs)
        return pages
    
if __name__ == '__main__':
    import os
    service = FileService()
    file_path = os.path.join("..", 'test', 'test_data', '台灣人壽新住院醫療保險附約.pdf')
    import pprint
    pprint.pprint(service.read_file(file_path)) # 方便閱讀
