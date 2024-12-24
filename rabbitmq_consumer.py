#!/usr/bin/env python
import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json

import pika

from dataflow_module.rabbitmq_task import QueueTaskDict
from dataflow_module.dataflow_service import dataflow_manager_instance
from graph_module.dto.simple_graph import SimpleGraph
from models.file_task import FileTask
from langchain_core.documents import Document

def main():
    rabbitmq_host = os.environ.get('RABBITMQ_HOST', 'localhost')
    rabbitmq_channel = os.environ.get('RABBITMQ_CHANNEL', 'llm_agent')
    rabbitmq_queue = os.environ.get('RABBITMQ_QUEUE', 'llm_agent')
    rabbitmq_user = os.environ.get('RABBITMQ_USER', None)
    rabbitmq_pass = os.environ.get('RABBITMQ_PASSWORD', None)
    credentials = None
    if rabbitmq_user and rabbitmq_pass:
        credentials = pika.PlainCredentials(rabbitmq_user, rabbitmq_pass)
    connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmq_host, credentials=credentials))
    channel = connection.channel()
    channel.queue_declare(queue=rabbitmq_channel)

    def callback(ch, method, properties, body):
        task: QueueTaskDict = json.loads(body)
        task_type = task['task_type']
        msg = task['msg']
        print(f" [x] Received {task_type}")
        if task_type == QueueTaskDict.TaskType.FILE_READ:
            msg = FileTask.load_from_json(msg)
            dataflow_manager_instance.received_file_task(msg)
        elif task_type == QueueTaskDict.TaskType.ENTITY_BUILD:
            msg = json.loads(msg)
            simple_graph: SimpleGraph = SimpleGraph.from_dict(msg)
            dataflow_manager_instance.received_entity_task(simple_graph)
        elif task_type == QueueTaskDict.TaskType.COMMNUITY_BUILD:
            dataflow_manager_instance.received_refine_task(task)
        elif task_type == QueueTaskDict.TaskType.RESTORE_NEO4J:
            dataflow_manager_instance.received_restore_neo4j(msg)
        elif task_type == QueueTaskDict.TaskType.BACKUP_NEO4J:
            dataflow_manager_instance.received_backup_neo4j()
        else:
            print(f" [x] Unknown task type: {task_type}")
            
        

    channel.basic_consume(queue=rabbitmq_queue,
                        auto_ack=True,
                        on_message_callback=callback)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)