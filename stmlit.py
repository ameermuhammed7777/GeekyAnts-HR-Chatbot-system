import streamlit as st
from backend_functions import functionality
import requests
import numpy as np

fn = functionality()

@st.cache_data(show_spinner=True)
def load_docs_and_index():
    docs = fn.textload("./utils/employee_table.txt")
    index = fn.build_faiss_index(docs)
    return docs, index

docs, index = load_docs_and_index()


st.title("GeekyAnts HR Chatbot👩‍💻 ")
user_query = st.text_input("Ask me something:")


if user_query:
    with st.spinner("🔎 Searching and generating answer..."):
        matched_docs = fn.doc_retrive(user_query, index, docs)
        response = fn.QandA(user_query, matched_docs)

    st.write(response)