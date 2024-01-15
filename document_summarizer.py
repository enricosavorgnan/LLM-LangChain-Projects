# ----------------------------------------------------------- #
# You may need to install the following libraries:            
# · langchain
# · langchain-community
# · langchain-openai
# · openai
# · faiss-cpu
# ----------------------------------------------------------- #

import os

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
  
# ------------------------------------------------------------------- #
# Initialising the environment where to move. Need an OpenAI API key  #
# ------------------------------------------------------------------- #

#initialising os environment with an openai key
openai_key = '      your_openai_api_key_goes_here      '
os.environ["OPENAI_API_KEY"] = openai_key


def create_vector_db(file_path: str) -> FAISS:

    loader = TextLoader(file)
    text = loader.load()

    embeddings = OpenAIEmbeddings()
        #chunking file into 200-words long chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 20)
    chunks = splitter.split_documents(text)

    return FAISS.from_documents(chunks, embeddings)
        #sending chunks to FAISS



# -------------------------------------------------------------------- #
# Now start working with LLM, langchain and the chunked file           #
# -------------------------------------------------------------------  #
def get_response(database_file):
    llm = OpenAI (temperature = 0.8, max_tokens = 500)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=database_file.as_retriever())
    query = str(input('Your query: '))

    response = []
    response = chain({'question': query})

    return response






file = input('Your file path: ')
database_file = create_vector_db(file)
get_response(database_file=database_file)
