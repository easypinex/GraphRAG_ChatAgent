import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    
from enum import Enum
from datetime import datetime
import json

from models.model_serialization import ModelSerialization
from sqlalchemy import Column, Integer, Unicode, DateTime
from sqlalchemy.types import Enum as SQLAlchemyEnum  # 使用別名以區分 SQLAlchemy 的 Enum
from sqlalchemy.ext.declarative import declarative_base
from models import Base  # 引用全局 Base


class FileTask(ModelSerialization, Base):
    class FileStatus(str, Enum):
        PENDING = "PENDING" # 檔案剛上傳, 準備建立Graph
        GRAPH_PROCESSING = "GRAPH_PROCESSING" # 正在建立基本 Graph: Document -> Parent -> Chunk
        GRAPH_ENTITY_PEDING = "GRAPH_ENTITY_PEDING" # 正在等待建立 Entity
        GRPAH_ENTITY = "GRPAH_ENTITY" # 正在建立實體圖
        REFINED_PENDING = "REFINED_PENDING" # 等待知識重整
        REFINING_KNOWLEDGE = "REFINING_KNOWLEDGE" # 知識重整中
        SUMMARIZING = "SUMMARIZING" # 總結知識中
        COMPLETED = "COMPLETED" # 完成
        FAILED = "FAILED" # 失敗
        DELETED = "DELETED" # 已經完整刪除

    class UserOperate(str, Enum):
        DELETE = "DELETE" # 刪除

    __tablename__ = 'file_tasks'
    
    id = Column(Integer, primary_key=True)
    # 檔案上傳 server 時的 temp folder path
    filedir = Column(Unicode(256), nullable=True)
    filename = Column(Unicode(256), nullable=False)
    status = Column(SQLAlchemyEnum(FileStatus), nullable=False, default=FileStatus.PENDING)
    last_updated = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)
    # 檔案上傳 minio 時的 folder path
    minio_dir = Column(Unicode(256), nullable=True)
    user_operate = Column(SQLAlchemyEnum(UserOperate), nullable=True, default=None)
    
    def to_dict(self):
        dict = super().to_dict()
        dict['last_updated'] = self.last_updated.isoformat()  # 格式化時間
        return dict
    
    @property
    def file_path(self):
        if self.filedir is not None:
            return os.path.join(self.filedir, self.filename)
        if self.minio_dir is not None:
            return os.path.join(self.minio_dir, self.filename)
        return self.filename

    @classmethod
    def load_from_dict(cls, data: dict):
        """Load an instance from a dictionary."""
        instance = cls(
            id=data.get('id'),
            filedir=data.get('filedir'),
            filename=data.get('filename'),
            status=cls.FileStatus[data['status']] if 'status' in data else cls.FileStatus.PENDING,
            last_updated=datetime.fromisoformat(data['last_updated']) if 'last_updated' in data else datetime.now()
        )
        return instance

    @classmethod
    def load_from_json(cls, json_data: str):
        """Load an instance from a JSON string."""
        data = json.loads(json_data)
        return cls.load_from_dict(data)

if __name__ == '__main__':
    task = file_task = FileTask(
        id=1,
        filedir="/uploads",
        filename="example.txt",
        status=FileTask.FileStatus.PENDING,
        last_updated=datetime.now()
    )
    json_str = json.dumps(task.to_dict(), indent=2)
    print(json_str)
    recover_task = FileTask.load_from_json(json_str)
    recover_task_str = json.dumps(recover_task.to_dict(), indent=2)
    print(recover_task_str)
    