from typing import TypedDict

class FileUploadSuccessDict(TypedDict):
    message : str
    file_ids: list[str]