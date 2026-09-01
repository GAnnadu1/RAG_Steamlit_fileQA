import os

import streamlit as st
from reg_utility import process_document_to_chroma_db, answer_question

#Set the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

st.title("✦ 📄 Document Question Answering 📄 ✦")

#file uploader widget
uploaded_file = st.file_uploader("Upload your PDF Document", type="pdf")

if uploaded_file is not None:
    #define save path
    file_path = os.path.join(working_dir, uploaded_file.name)
    #Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write(f"Processing {uploaded_file.name}...")
    process_document_to_chroma_db(uploaded_file.name)
    st.write("Document processing complete.")

# Question input
user_question = st.text_input("Ask a question about the document:")

if user_question:
    if 'db' not in st.session_state:
        st.error("Please upload and process a document first.")
    else:
        st.write("Generating response...")
        response = answer_question(user_question)
        st.write(response)
