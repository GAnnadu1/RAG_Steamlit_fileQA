import os
import streamlit as st
from reg_utility import process_document_to_chroma_db, answer_question

# Set the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(layout="wide")

st.title("✦ 📄 GenAI Question Answering ✦")

# File uploader widget
uploaded_file = st.file_uploader("Upload your PDF Document", type="pdf")

if uploaded_file is not None:
    # Define save path
    save_path = os.path.join(
        working_dir,
        uploaded_file.name
    )
    # Save the file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    process_document = process_document_to_chroma_db(uploaded_file.name)
    st.info("Document processing Successfully.")

# Question input
user_question = st.text_area("Ask a question about the document:")

if st.button("Answer"):
    answer, sources = answer_question(user_question)

    st.markdown("### GenAI Response")
    st.markdown(answer)
    st.info(f"Response generated successfully from source: {', '.join(sources)}.")
