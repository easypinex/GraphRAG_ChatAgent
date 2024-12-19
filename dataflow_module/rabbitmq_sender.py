import pika
import os
import json

from dataflow_module.rabbitmq_task import QueueTaskDict

# RabbitMQ configuration
rabbitmq_host = os.environ.get('RABBITMQ_HOST', 'localhost')
rabbitmq_queue = os.environ.get('RABBITMQ_QUEUE', 'llm_agent')
rabbitmq_channel = os.environ.get('RABBITMQ_CHANNEL', 'llm_agent')

def publish_queue_message(task: QueueTaskDict):
    try:
        # Establish a connection to RabbitMQ
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host))
        channel = connection.channel()
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