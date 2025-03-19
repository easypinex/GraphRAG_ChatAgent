# 標準庫
import warnings
import re
import sys
import os
import csv
import ast
import io
import time
import json
import pickle
import hashlib
import inspect
import tiktoken
import tiktoken_ext.openai_public
from pprint import pprint

# 第三方庫
import jieba
import difflib                
import torch
import pdfplumber
import pandas as pd
pd.set_option('display.max_rows', None)  # 顯示所有列
pd.set_option('display.max_columns', None)  # 顯示所有欄位
pd.set_option('display.width', None)  # 自適應寬度
pd.set_option('display.max_colwidth', None)  # 不截斷內容
import numpy as np
import streamlit as st
from summarizer import Summarizer
from dotenv import load_dotenv
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz   

from collections import defaultdict, OrderedDict, Counter

import matplotlib.pyplot as plt
import matplotlib
import pyLDAvis
import pyLDAvis.gensim_models as pyLDAvis_gensim_models
import textwrap
from keybert import KeyBERT
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
from textrank4zh import TextRank4Keyword, TextRank4Sentence

# LangChain 相關
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQA, GraphCypherQAChain, RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_redis import RedisChatMessageHistory

# 機器學習相關
import gensim
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from pprint import pprint
from numba import jit, cuda 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
