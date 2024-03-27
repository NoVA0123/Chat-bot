# mocel path
model_path_hub = 'TheBloke/Llama-2-7B-Chat-GGML'
model_basename = 'llama-2-7b-chat.ggmlv3.q2_K.bin'
model_path = '/Chat-bot'


# importing libraries
import os
from huggingface_hub import hf_hub_download

def download(model_path_hub, model_basename):
    cur_path = os.getcwd() +'/model'
    if os.path.exists(model_basename):
        print('model already exists')
        return cur_path
    else:
       return hf_hub_download(repo_id=model_path_hub,
                              filename=model_basename,
                              cache_dir=cur_path) 


directory = download(model_path_hub,
                     model_basename)
print(directory)