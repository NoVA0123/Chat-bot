from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


# pdf loader function
def LoadPdf(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    return documents


# Text splitter
def TextSplit(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Embedding downloader
def DownloadEmbedding():
    cur_path = os.getcwd() + '/model'
    if os.path.exists(cur_path+'/huggingface'):
        print('FIle already exis')
        return cur_path+'/huggingface'
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       cache_folder=cur_path)
    return embeddings
