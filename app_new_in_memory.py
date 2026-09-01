import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Function to load CSS
def load_css(css_path):
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the custom CSS
load_css(os.path.join(working_dir, "style.css"))

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="In-Memory RAG App")
st.title("🧠 In-Memory RAG with ChromaDB 🧠")

# Initialize LLM and embedding model (cached to prevent re-initialization)
@st.cache_resource
def initialize_models():
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not found in .env file. Please set it.")
        st.stop()
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, google_api_key=GEMINI_API_KEY)
    embedding = HuggingFaceEmbeddings()
    return llm, embedding

llm, embedding = initialize_models()

# Function to process document and create in-memory Chroma DB
def process_document_to_in_memory_chroma_db(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    # Create in-memory Chroma vector database (no persist_directory)
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding
    )
    return vectordb

# Streamlit UI
uploaded_file = st.file_uploader("Upload your PDF Document for In-Memory Processing", type="pdf")

if uploaded_file is not None:
    # Save the file temporarily
    working_dir = os.path.dirname(os.path.abspath(__file__))
    temp_file_path = os.path.join(working_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document and building in-memory vector store..."):
        vectordb = process_document_to_in_memory_chroma_db(temp_file_path)
        st.session_state['in_memory_vectordb'] = vectordb
    st.success("Document processed and in-memory vector store built successfully!")
    os.remove(temp_file_path) # Clean up temp file

user_question = st.text_area("Ask a question about the uploaded document (In-Memory):")

if st.button("Get Answer (In-Memory)"):
    if 'in_memory_vectordb' not in st.session_state:
        st.warning("Please upload and process a document first.")
    elif user_question:
        with st.spinner("Generating response..."):
            retriever = st.session_state['in_memory_vectordb'].as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            response = qa_chain.invoke({"query": user_question})
            answer = response["result"]
            source_documents = response["source_documents"]
            source_names = list(set([os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in source_documents]))

            st.markdown("### GenAI Response (In-Memory)")
            st.markdown(answer)
            if source_names:
                st.info(f"Sources: {', '.join(source_names)}")
            else:
                st.info("No specific sources found.")
    else:
        st.warning("Please enter a question.")
