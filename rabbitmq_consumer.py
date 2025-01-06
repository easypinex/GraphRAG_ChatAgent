#!/usr/bin/env python
import os
import sys
import json

from dataflow_module.rabbitmq_task import QueueTaskDict
from dataflow_module.dataflow_service import dataflow_manager_instance
from graph_module.dto.simple_graph import SimpleGraph
from models.file_task import FileTask
from dataflow_module.rabbitmq_connection import get_rabbitmq_connection
from logger.logger import get_logger

logging = get_logger()
def main():
    rabbitmq_channel = os.environ.get('RABBITMQ_CHANNEL', 'llm_agent')
    rabbitmq_queue = os.environ.get('RABBITMQ_QUEUE', 'llm_agent')
    connection = get_rabbitmq_connection()
    channel = connection.channel()
    channel.queue_declare(queue=rabbitmq_channel)

    def callback(ch, method, properties, body):
        task: QueueTaskDict = json.loads(body)
        dataflow_manager_instance.received_mq_task(task)
        

    channel.basic_consume(queue=rabbitmq_queue,
                        auto_ack=True,
                        on_message_callback=callback)

    logging.info(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)