import sys
import os
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    
import json
import traceback

from dataflow_module.rabbitmq_connection import get_rabbitmq_connection
from dataflow_module.rabbitmq_task import QueueTaskDict
from logger.logger import get_logger

logging = get_logger()

def publish_queue_message(task: QueueTaskDict):
    try:
        # Establish a connection to RabbitMQ
        connection = get_rabbitmq_connection()
        channel = connection.channel()
        rabbitmq_queue = os.environ.get('RABBITMQ_QUEUE', 'llm_agent')
        # Declare the queue, auto create queue
        channel.queue_declare(queue=rabbitmq_queue)
        message = json.dumps(task)
        # Publish the message
        channel.basic_publish(exchange='', routing_key=rabbitmq_queue, body=message)
        connection.close()
        return True
    except:
        logging.error(f"Failed to send message: {traceback.format_exc()}")
        return False