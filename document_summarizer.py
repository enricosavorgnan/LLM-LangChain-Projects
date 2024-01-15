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

#initialising the file
file = input('Your file path: ')
loader = TextLoader(file)
text = loader.load()

#chunking file into 200-words long chunks
embeddings = OpenAIEmbeddings()
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20)
chunks = splitter.split_documents(text)

#sending chunks to FAISS
index_vector = FAISS.from_documents(chunks, embeddings)


# -------------------------------------------------------------------- #
# Now start working with LLM, langchain and the chunked file           #
# -------------------------------------------------------------------  #

llm = OpenAI (temperature = 0.8, max_tokens = 500)
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=index_vector.as_retriever())
query = str(input('Your query: '))

response = []
response = chain({'question': query})