import pika
import os
import json

from dataflow_module.rabbitmq_connection import get_rabbitmq_connection
from dataflow_module.rabbitmq_task import QueueTaskDict

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
    except Exception as e:
        print(f"Failed to send message: {e}")
        return False