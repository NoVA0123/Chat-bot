from src.helper import LoadPdf, TextSplit, DownloadEmbedding
from langchain_community.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os


# Importing keys from .env file(You have to create your own from Pinecone)
load_dotenv()
PineconeApiKey = os.environ.get('PineconeApiKey')
PineconeApiEnv = os.environ.get('PineconeApiEnv')


print(PineconeApiKey)
