# importing libraries
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
import os


# Pinecone api key

PineconeApiKey = '74f7589d-ddff-41d3-b65e-e7489ead58fb'
PineconeApiEnv = 'gcp-starter'


# function to load text
def load_pdf(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    return documents

# load the data
extracted_data = load_pdf('data/')


# Text Splitter function
def TextSplit(extraced_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Splitting the data into chunks
text_chunk = TextSplit(extracted_data)
# print(len(text_chunk))


# Embedding downloader
def DownloadEmbedding():
    cur_path = os.getcwd + '/model'
    if os.path.exists(cur_path+'/huggingface'):
        print('FIle already exis')
        return cur_path+'/huggingface'
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       cache_folder=cur_path)
    return embeddings

# Downloading the embedding model
embeddings = DownloadEmbedding()
# print(embeddings)


# Initializing pinecone
pinecone.init(api_key=PineconeAPiKey,
              environment=PineconeApiEnv)

index_name = 'chatbot'

# Creating embeddings for each text
DocumentSearch = Pinecone.from_texts([tmp.page_conetent for tmp in text_chunks],
                                     embeddings=embeddings,
                                     index_name=index_name)


# Creating a prompt
prompt_template='''
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that I don't know the answer,
don't try to make it up.

Context: {context}
question: {question}

Only return the helpful and direct  answer and nothing else
Helpful answer:
'''


# generating a template
PROMPT = PromptTemplate(template=prompt_template,
                        input_variables=['content', 'question'])
chain_type_kwargs={'pronpt': PROMPT}


# Bringing transformer to use the model
llm = CTransformers(model='model/llama-2-7b-chat.ggmlv3.q2_K.bin',
                    model_type='llama',
                    config={'max_new_tokens':512,
                            'temprature':0.9})


# The actual model

QA = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=docsearch.as_retriever(search_kwargs={'k':2}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
        )
