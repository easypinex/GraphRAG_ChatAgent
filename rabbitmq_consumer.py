import asyncio
import json
import os
from dataflow_module.rabbitmq_connection import get_rabbitmq_connection
from dataflow_module.rabbitmq_task import QueueTaskDict
from dataflow_module.dataflow_service import dataflow_manager_instance
from graph_module.dto.simple_graph import SimpleGraph
from models.file_task import FileTask

async def process_message(message):
    async with message.process():
        try:
            task: QueueTaskDict = json.loads(message.body.decode())
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
        except Exception as e:
            print(f"Error processing task: {e}")

async def main():
    rabbitmq_url = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost/")
    rabbitmq_queue = os.environ.get('RABBITMQ_QUEUE', 'llm_agent')

    connection = await get_rabbitmq_connection()

    async with connection:
        channel = await connection.channel()

        # Enable QoS (Quality of Service)
        await channel.set_qos(prefetch_count=1)

        # Declare queue
        queue = await channel.declare_queue(rabbitmq_queue, durable=True)

        print(" [*] Waiting for messages. To exit press CTRL+C")

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                await process_message(message)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted")