import os
from aio_pika import connect_robust
import ssl

async def get_rabbitmq_connection():
    rabbitmq_host = os.environ.get('RABBITMQ_HOST', 'localhost')
    rabbitmq_port = int(os.environ.get('RABBITMQ_PORT', 5672))
    rabbitmq_user = os.environ.get('RABBITMQ_USER', None)
    rabbitmq_pass = os.environ.get('RABBITMQ_PASSWORD', None)
    rabbitmq_cert_verify = os.getenv("RABBITMQ_CERT_VERIFY", "true").lower() == "true"
    rabbitmq_ssl = os.getenv("RABBITMQ_SSL", "true").lower() == "true"
    ssl_context = None

    if rabbitmq_ssl:
        ssl_context = ssl.create_default_context()
        if not rabbitmq_cert_verify:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

    connection = await connect_robust(
        host=rabbitmq_host,
        port=rabbitmq_port,
        login=rabbitmq_user,
        password=rabbitmq_pass,
        ssl=rabbitmq_ssl,
        ssl_context=ssl_context if rabbitmq_ssl else None
    )

    return connection