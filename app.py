from pathlib import Path
from flask import Flask, render_template, jsonify, request
from src.helper import DownloadEmbedding
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from llama_cpp import Llama


# Importing embedding model
embeddings = DownloadEmbedding()


# Model Download
model_path_hub = 'TheBloke/Llama-2-7B-Chat-GGML'
model_basename = 'llama-2-7b-chat.ggmlv3.q4_0.bin'

# importing libraries
import os
from huggingface_hub import hf_hub_download


def download(model_path_hub, model_basename):
    cur_path = os.getcwd() +'/model'
    if os.path.exists(cur_path+model_basename):
        print('model already exists')
        return cur_path
    else:
        return hf_hub_download(repo_id=model_path_hub,
                               filename=model_basename,
                               cache_dir=cur_path) 
directory = download(model_path_hub, model_basename)


# applying flask
app = Flask(__name__)

# Loading Pinecone api
# Importing keys from .env file(You have to create your own from Pinecone)
load_dotenv()
PineconeApiKey = os.environ.get('PineconeApiKey')
PineconeApiEnv = os.environ.get('PineconeApiEnv')


# Initializing Pinecone
pinecone.init(api_key=PineconeApiKey,
              environment=PineconeApiEnv)

index_name = 'chatbot'

# Load index from existing index
docsearch = Pinecone.from_existing_index(index_name, embeddings)


# generating a template
PROMPT = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question'])
chain_type_kwargs={'prompt': PROMPT}


# Bringing transformer to use the model
model_path = Path(directory)
llm=CTransformers(model_file=model_path,
                  #model_type="llama"
                  config={'max_new_tokens':512,
                          'temperature':0.8})
'''llm = Llama(model_path='model\llama-2-7b-chat.Q2_K.gguf',
               n_threads=4,
               n_batch=512,
               n_gpu_layers=32)
'''

# The actual model
QA = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=docsearch.as_retriever(search_kwargs={'k':2}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
        )


# deafault route
@app.route('/')
def index():
    return render_template('chat.html')


# Final route
@app.route('get', methods=['GET', 'POST'])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)
    result = QA({'query': input})
    print('Response: ', result['result'])
    return str(result['result'])



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
