from enum import Enum
from typing import TypedDict
import json
from datetime import datetime
class QueueTaskDict(TypedDict):
    class TaskType(str, Enum):
        FILE_READ = "FILE_READ"
        ENTITY_BUILD = "ENTITY_BUILD"
        COMMNUITY_BUILD = "COMMNUITY_BUILD"
        RESTORE_NEO4J = "RESTORE_NEO4J"
        BACKUP_NEO4J = "BACKUP_NEO4J"
    task_type: TaskType
    msg: str
    send_datetime: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def create_queue_task(**kwargs) -> 'QueueTaskDict':
        task = QueueTaskDict(**kwargs)
        if task.get("send_datetime") is None:
            task['send_datetime'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        return task
    
if __name__ == "__main__":
    task = QueueTaskDict.create_queue_task(task_type=QueueTaskDict.TaskType.FILE_READ, msg="test")
    print(json.dumps(task, indent=2))
    