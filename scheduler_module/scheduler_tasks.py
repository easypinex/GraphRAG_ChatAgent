import sys
import os

from dataflow_module.rabbitmq_sender import publish_queue_message
from dataflow_module.rabbitmq_task import QueueTaskDict

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from logger.logger import get_logger

logging = get_logger()

def daily_task():
    logging.info('start daily_task!')
    # 實際的排程邏輯
    publish_queue_message(QueueTaskDict.create_queue_task(task_type=QueueTaskDict.TaskType.COMMNUITY_BUILD, msg=None))
