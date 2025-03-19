# 標準庫
import ast
import csv
import hashlib
import inspect
import io
import json
import os
import pickle
import re
import time
import tiktoken
import tiktoken_ext.openai_public
import warnings

# 第三方庫
import difflib
import jieba
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfplumber
import pyLDAvis.gensim_models
import redis
import torch
from collections import Counter, OrderedDict, defaultdict
from difflib import SequenceMatcher
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from keybert import KeyBERT
from summarizer import Summarizer

# CKIP相關
from ckip_transformers.nlp import CkipNerChunker, CkipPosTagger, CkipWordSegmenter

# LangChain 相關
from langchain.chains import GraphCypherQAChain, RetrievalQA, RetrievalQAWithSourcesChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Neo4jVector
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI
from langchain_redis import RedisChatMessageHistory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# 機器學習相關
import gensim
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from numba import cuda, jit
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
