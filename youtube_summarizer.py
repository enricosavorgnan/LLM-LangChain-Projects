# ----------------------------------------------------------- #
# You may need to install the following libraries:            
# · langchain
# · langchain-community
# · langchain-openai
# · openai
# · faiss-cpu
# ----------------------------------------------------------- #

import os
from langchain.llms import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter



# ------------------------------------------------------------------- #
# Initialising the environment where to move. Need an OpenAI API key  #
# ------------------------------------------------------------------- #

openai_key = "      your_openai_api_key_goes_here      "
os.environ["OPENAI_API_KEY"] = openai_key



def create_vector_db(video_url: str) -> FAISS:
# Creates a vector database from a YouTube video URL
    
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=100)
        #splits the transcript into chunks of 1000 characters and 100 characters overlap
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
        #db stands for database. 
    return db


# -------------------------------------------------------------------- #
# Now start working with LLM, langchain and the chunked file           #
# -------------------------------------------------------------------  #

def get_response(db, query, k=4):
        #k = 4 because OpenAI manages 4*1000 token
    docs = db.similarity_search(query)
    docs_page_content = ''.join([d.page_content for d in docs])

    template = """
    You are a Youtube Assistant that is able to answer questions about a specific video.
    Answer the  following question: {question}
    Search the following video transcript: {docs}
    Use only factual information from the transcript. Your answer should be detailed.
    If you do not have enough infomration to answer, simply say "I don't know".
    """

    llm = OpenAI(model_name='text-davinci-003')

    prompt = PromptTemplate(
        input_variables= ['question', 'docs'],
        template = template,
    )    

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question = query, docs = docs_page_content)
    response = response.replace('\n', '')

    return response



embeddings = OpenAIEmbeddings()
    #limite, "embedding", di OpenAI è circa 4000 token per ogni input
video_url = str(input("Your video url: "))
query = str(input("Your query: "))

db = create_vector_db (video_url)
print(get_response(db, query))