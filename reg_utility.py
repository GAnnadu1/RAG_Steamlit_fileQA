#Dependency
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA

#Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')

#Add Directory 
working_dir = os.path.dirname(os.path.abspath(__file__))

#Load the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Load llm
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

#Document injection function
def process_document_to_chroma_db(file_name):
    #Load the PDF document using UnstructuredPDFLoader
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()

    #Split the text into chunks for emberdding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    #Store the document chunks in a Chroma vector database
    vectordb = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings, 
        persist_directory=f"{working_dir}/doc_vectorstore"
    )
    return 0


#Doucment Question and answering function
def answer_question(user_question):
  #Load the persistant Chroma vector database
  vectordb = Chroma(
      persist_directory=f"{working_dir}/doc_vectorstore", 
      embedding_function=embeddings
  )
  
  #Create a retriever for document search
  retriever = vectordb.as_retriever()

  #Create a RetrievalQA chain to answer user questions using GenAI
  qa_chain = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=retriever,
  )
  response = qa_chain.invoke({"query": user_question})
  answer = response["result"]

  return answer
