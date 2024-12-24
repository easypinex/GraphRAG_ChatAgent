import asyncio
from aio_pika import DeliveryMode, Message
import os
import json

from dataflow_module.rabbitmq_connection import get_rabbitmq_connection
from dataflow_module.rabbitmq_task import QueueTaskDict


def publish_queue_message_sync(task):
    # 使用 asyncio.run 調用異步函数
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(publish_queue_message(task))
    loop.close()
    return result

async def publish_queue_message(task):
    try:
        # Establish a connection to RabbitMQ
        connection = await get_rabbitmq_connection()
        async with connection.channel() as channel:
            rabbitmq_queue = os.environ.get('RABBITMQ_QUEUE', 'llm_agent')

            # Declare the queue, auto create queue
            await channel.declare_queue(rabbitmq_queue, durable=True)

            message = json.dumps(task)

            # Publish the message
            await channel.default_exchange.publish(
                Message(
                    body=message.encode(),
                    delivery_mode=DeliveryMode.PERSISTENT
                ),
                routing_key=rabbitmq_queue
            )

        await connection.close()
        return True

    except Exception as e:
        print(f"Failed to send message: {e}")
        return False