from src.helper import LoadPdf, TextSplit, DownloadEmbedding
from langchain_community.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os


# Importing keys from .env file(You have to create your own from Pinecone)
load_dotenv()
PineconeApiKey = os.environ.get('PineconeApiKey')
PineconeApiEnv = os.environ.get('PineconeApiEnv')
# print(PineconeApiKey)
# print(PineconeApiENv)


# loading pdf
extracted_data = LoadPdf('data/')
# splitting it into chunks
text_chunks = TextSplit(extracted_data)
# downloading embedding model
embeddings = DownloadEmbedding()


# Initializing Pinecone
pinecone.init(api_key=PineconeApiKey,
              environment=PineconeApiEnv)

index_name = 'chatbot'

# using pinecone to convert text into embeddings
DocumentSearch = Pinecone.from_texts([tmp.page_content for tmp in text_chunks],
                                     embedding=embeddings,
                                     index_name=index_name)
