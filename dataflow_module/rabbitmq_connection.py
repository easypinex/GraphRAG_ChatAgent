import os
import pika
import ssl

def get_rabbitmq_connection():
    rabbitmq_host = os.environ.get('RABBITMQ_HOST', 'localhost')
    rabbitmq_port = int(os.environ.get('RABBITMQ_PORT', 5672))
    rabbitmq_user = os.environ.get('RABBITMQ_USER', None)
    rabbitmq_pass = os.environ.get('RABBITMQ_PASSWORD', None)
    rabbitmq_cert_verify = os.getenv("RABBITMQ_CERT_VERIFY", "true").lower() == "true"
    rabbitmq_ssl = os.getenv("RABBITMQ_SSL", "true").lower() == "true"
    ssl_options = None
    if rabbitmq_ssl:
        ssl_context = ssl.create_default_context()
        if not rabbitmq_cert_verify:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        ssl_options = pika.SSLOptions(context=ssl_context)
    credentials = None
    if rabbitmq_user and rabbitmq_pass:
        credentials = pika.PlainCredentials(rabbitmq_user, rabbitmq_pass)
    connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmq_host, rabbitmq_port, credentials=credentials, ssl_options=ssl_options, heartbeat=0))
    return connection