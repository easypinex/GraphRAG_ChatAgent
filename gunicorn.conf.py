import os
bind = "0.0.0.0:6000"
certfile = os.getenv("CHAT_AGENT_SSL_PUBLIC_KEY")
keyfile = os.getenv("CHAT_AGENT_SSL_PRIVATE_KEY")
workers = 1