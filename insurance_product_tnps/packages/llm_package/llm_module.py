
from langchain_ollama import OllamaLLM
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from .llm_config import *

OLLAMA_MODEL = "qwen2-72b-instruct"
LLM_QWEN = OllamaLLM(model=OLLAMA_MODEL, temperature=0.0)

LLM_4O = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            model_name=AZURE_OPENAI_DEPLOYMENT_NAME,
            openai_api_key=AZURE_OPENAI_API_KEY,
            temperature=0.3,
            streaming=True,
        )

EMB_MODEL = AzureOpenAIEmbeddings(
    model=AZURE_EMB_MODLE,
    azure_deployment=AZURE_EMB_DEPLOYMENT,
    azure_endpoint=AZURE_EMB_ENDPOINT,
    openai_api_version=AZURE_EMB_API_VERSION,
    api_key=AZURE_EMB_MODLE_API_KEY,
)

def generate_response_for_query(chain, query_params):
    ret_str = ""
    for char in chain.stream(query_params):  # 從鏈中流式生成查詢的響應
        if hasattr(char, 'content'):
            # Azure 的回應
            ret_str += char.content  # 將內容添加到主題句子
        else:  # 如果沒有內容屬性
            # Ollama 的回應
            ret_str += str(char)  # 將回應轉為字串並添加到主題句子
    return ret_str
