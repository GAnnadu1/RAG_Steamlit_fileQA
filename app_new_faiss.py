import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="FAISS RAG App")
st.title("🚀 RAG with FAISS Vector Store 🚀")

# Initialize LLM and embedding model (cached to prevent re-initialization)
@st.cache_resource
def initialize_models_faiss():
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not found in .env file. Please set it.")
        st.stop()
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, google_api_key=GEMINI_API_KEY)
    embedding = HuggingFaceEmbeddings()
    return llm, embedding

llm_faiss, embedding_faiss = initialize_models_faiss()

# Function to process document and create FAISS index
def process_document_to_faiss_db(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    # Create FAISS vector database from documents
    faiss_db = FAISS.from_documents(
        documents=texts,
        embedding=embedding_faiss
    )
    return faiss_db

# Streamlit UI
uploaded_file_faiss = st.file_uploader("Upload your PDF Document for FAISS Processing", type="pdf")

if uploaded_file_faiss is not None:
    # Save the file temporarily
    working_dir_faiss = os.path.dirname(os.path.abspath(__file__))
    temp_file_path_faiss = os.path.join(working_dir_faiss, uploaded_file_faiss.name)
    with open(temp_file_path_faiss, "wb") as f:
        f.write(uploaded_file_faiss.getbuffer())

    with st.spinner("Processing document and building FAISS index..."):
        faiss_db = process_document_to_faiss_db(temp_file_path_faiss)
        st.session_state['faiss_db'] = faiss_db
    st.success("Document processed and FAISS index built successfully!")
    os.remove(temp_file_path_faiss) # Clean up temp file

user_question_faiss = st.text_area("Ask a question about the uploaded document (FAISS):")

if st.button("Get Answer (FAISS)"):
    if 'faiss_db' not in st.session_state:
        st.warning("Please upload and process a document first.")
    elif user_question_faiss:
        with st.spinner("Generating response..."):
            retriever_faiss = st.session_state['faiss_db'].as_retriever()
            qa_chain_faiss = RetrievalQA.from_chain_type(
                llm=llm_faiss,
                chain_type="stuff",
                retriever=retriever_faiss,
                return_source_documents=True
            )
            response_faiss = qa_chain_faiss.invoke({"query": user_question_faiss})
            answer_faiss = response_faiss["result"]
            source_documents_faiss = response_faiss["source_documents"]
            source_names_faiss = list(set([os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in source_documents_faiss]))

            st.markdown("### GenAI Response (FAISS)")
            st.markdown(answer_faiss)
            if source_names_faiss:
                st.info(f"Sources: {', '.join(source_names_faiss)}")
            else:
                st.info("No specific sources found.")
    else:
        st.warning("Please enter a question.")
