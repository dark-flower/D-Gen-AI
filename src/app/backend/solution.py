
# Use LangChain to do apply the RAG. We will need:
# 
# 1. Load the long document and a chunker that segments the document into N overlapping segments.
# 2. Use Embeddings from an encoder model. We use Microsoft's MiniLM which is fast and has very good performance.
# 3. Use in-memory vector database FAISS which stores the embeddings of the text and retrieves the data based on similarity.
# 4. Use the RetrievalQA chain to make the model answer user's questions


import os
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings as TextEmbeddings
from langchain.vectorstores import FAISS as faiss_vdb

from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain

from backend.configs import *

import argostranslate.package
import argostranslate.translate
from mtranslate import translate


# ## Data Preparation


# below are the experiment configurations. The performance of the solution depends on:
# 
# 1. Chunk size: you may need to change it a couple of times before you get the best results.
# 2. Chunk overlap: we do overlap to avoid incoherent chunks
# 3. model type: please use Mistral for best and fastest retrieval capabilities, or llama for chat capabilities.




# load the text documents
text_loader_kwargs={'autodetect_encoding': True}
data_loader = DirectoryLoader(FILES_DIR, glob="*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
text_chunker = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZES_CHARS, chunk_overlap=CHUNKS_OVERLAP_CHARS)

documents = data_loader.load()
chunks = text_chunker.split_documents(documents)


embedder = TextEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", #"microsoft/Multilingual-MiniLM-L12-H384",
    model_kwargs={'device': DEVICE})



try:
    # do not recreate the database every time, just load it
    db_hander = faiss_vdb.load_local(DB_NAME)
except:
    db_handler = faiss_vdb.from_documents(chunks, embedder)
    db_handler.save_local(DB_NAME)



# ## Chat Model and Prompt Preparation


# Download the model in GGUF format: this is a compressed format and runnable in C++ to make it as fast as possible


if not os.path.exists(MODELS_PATH):
    os.mkdir(MODELS_PATH)

if MODEL_TYPE == "MISTRAL":
  model_key = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
  if not os.path.exists(f"{MODELS_PATH}{model_key}"):
    os.system(f"huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf  --local-dir {MODELS_PATH} --local-dir-use-symlinks False")

elif MODEL_TYPE == "LLAMA":
  model_key = "firefly-llama2-13b-chat.Q4_0.gguf"
  if not os.path.exists(f"{MODELS_PATH}{model_key}"):
    os.system(f"huggingface-cli download TheBloke/firefly-llama2-13B-chat-GGUF firefly-llama2-13b-chat.Q4_0.gguf  --local-dir {MODELS_PATH} --local-dir-use-symlinks False")

else:
  raise Exception(f"Unsupported model type: {MODEL_TYPE}. Supported types are: [ `MISTRAL`,  `LLAMA` ]")



# load the language model

#! Note: if you get this error:
# OSError: /lib64/libm.so.6: version `GLIBC_2.29' not found (required by ~/lib/python3.12/site-packages/ctransformers/lib/avx2/libctransformers.so)
#! please follow the steps in this answer to solve it: https://stackoverflow.com/a/75630806

language_model = CTransformers(model=f"{MODELS_PATH}{model_key}",
                    model_type='llama',
                    config={'max_new_tokens': MAX_GENERATION_NEW_TOKENS, 'temperature': GENERATION_TEMP},
                    n_gpu_layers=15
                  )



# Tell the database that we need it to work as retriever. It uses similarity measures for high dimensional vectors, and it returns top 3 results. These results will be given to the LLM to help it answer the user's questions.


db_retriever = db_handler.as_retriever(search_kwargs={'k': 3, 'search_type':'similarity'})



# prepare the prmpt we will use on the language model.
# We are using minimal template to save time and speed-up the model inference.

prompt_format = """Use the following pieces of information to answer the user's question. If the question is in Arabic, respond in Arabic.
If you don't know the answer just say 'Information is NOT available' and don't try to make up an answer.
Context: {context}
Question: {question}?
Only return the answer below and nothing else.
Answer:
"""


prompt_template = PromptTemplate(
    template=prompt_format,
    input_variables=['context', 'question'])


qa_lm = RetrievalQA.from_chain_type(llm=language_model,
                                     chain_type='stuff',
                                     retriever=db_retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={'prompt': prompt_template})



from_code = "en"
to_code = "ar"

argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    filter(
        lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
    )
)
argostranslate.package.install_from_path(package_to_install.download())

argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    filter(
        lambda x: x.from_code == to_code and x.to_code == from_code, available_packages
    )
)
argostranslate.package.install_from_path(package_to_install.download())


def answer_user_question(question:str="this is a dummy question, don't answer it", source_lang:str='en'):
    
    user_prompt = question
    if source_lang == 'ar':
        user_prompt = translate(question, 'en')

    output = qa_lm({'query': user_prompt})

    if source_lang == 'ar':
        output['response'] = translate(output["result"], 'ar')
    else:
       output['response'] = output['result']

    return output



