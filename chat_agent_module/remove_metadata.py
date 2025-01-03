from langchain.schema.runnable import Runnable
from langchain.schema.document import Document

class RemoveMetadata(Runnable):
    def invoke(self, intputs: list[Document], config = None, *args, **kwargs) -> list[Document]:
        for doc in intputs:
            doc.metadata = {}
        return intputs