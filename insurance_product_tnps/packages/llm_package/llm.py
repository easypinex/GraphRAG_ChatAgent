
from langchain_ollama import OllamaLLM
from langchain_openai import AzureChatOpenAI

from .llm_config import *

OLLAMA_MODEL = "qwen2-72b-instruct"
llm_qwen = OllamaLLM(model=OLLAMA_MODEL, temperature=0.0)

llm_4o = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            model_name=AZURE_OPENAI_DEPLOYMENT_NAME,
            openai_api_key=AZURE_OPENAI_API_KEY,
            temperature=0.3,
            streaming=True,
        )
