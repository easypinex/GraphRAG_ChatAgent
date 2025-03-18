import re
import os
import csv
import ast
import io
import time
import json
import pickle

import redis
import jieba
import difflib                
import torch
import pdfplumber
import pandas as pd
import numpy as np
import streamlit as st
from summarizer import Summarizer
from dotenv import load_dotenv
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz   

from collections import defaultdict
from collections import OrderedDict
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib
import pyLDAvis.gensim_models

import textwrap
from keybert import KeyBERT
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQA
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker

import gensim
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from pprint import pprint
from numba import jit, cuda 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI

from langchain_community.vectorstores import Neo4jVector
from langchain_openai import AzureOpenAIEmbeddings

from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_redis import RedisChatMessageHistory

import hashlib
import tiktoken
import tiktoken_ext.openai_public
import inspect

import warnings


__all__ = [
"re",
"os",
"csv",
"ast",
"io",
"time",
"json",
"pickle",
"jieba",
"difflib",            
"torch",
"pdfplumber",
"pd",
"np",
"st",
"Summarizer",
"load_dotenv",
"SequenceMatcher",
"fuzz"  ,
"defaultdict",
"OrderedDict",
"Counter",
"plt",
"matplotlib",

"textwrap",
"KeyBERT",

"ChatPromptTemplate",
"Neo4jGraph",
"PyPDFLoader",
"HuggingFaceEmbeddings",

"Neo4jVector",
"RetrievalQA",
"GraphCypherQAChain",
"PromptTemplate",
"RetrievalQAWithSourcesChain",

"RecursiveCharacterTextSplitter",
"CharacterTextSplitter",
"ChatPromptTemplate",
"MessagesPlaceholder",
"CkipWordSegmenter",
"CkipPosTagger",
"CkipNerChunker",

"gensim",
"corpora",
"models",
"CoherenceModel",
"LdaModel",
"pprint",
"jit",
"cuda", 

"TfidfVectorizer",
"cosine_similarity",

"ChatPromptTemplate",
"HumanMessage" ,
"SystemMessage",
"AIMessage",
"AzureChatOpenAI",

"Neo4jVector",
"AzureOpenAIEmbeddings",

"BaseChatMessageHistory",
"AIMessage",
"HumanMessage",
"ChatPromptTemplate",
"MessagesPlaceholder",
"RunnableWithMessageHistory",
"ChatOpenAI",
"RedisChatMessageHistory",

"hashlib",
"tiktoken",
"inspect",
"warnings",
"redis",

]