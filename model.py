# importing libraries
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers


# Pinecone api key

PineconeApiKey = '74f7589d-ddff-41d3-b65e-e7489ead58fb'
PineconeApiEnv = 'gcp-starter'


# function to load text
def load_pdf(data):
    loader = DIrectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    return documents
