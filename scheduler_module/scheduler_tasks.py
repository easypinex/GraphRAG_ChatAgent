import sys
import os

from dataflow_module.rabbitmq_sender import publish_queue_message_sync
from dataflow_module.rabbitmq_task import QueueTaskDict

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def daily_task():
    print('start daily_task!')
    # 實際的排程邏輯
    publish_queue_message_sync(QueueTaskDict.create_queue_task(task_type=QueueTaskDict.TaskType.COMMNUITY_BUILD, msg=None))
