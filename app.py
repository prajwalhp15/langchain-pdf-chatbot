import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import tempfile
import os

# Streamlit UI
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("🧠 PDF Chatbot with LangChain + HuggingFace")

# File upload
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load and process PDF
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)

    hf_pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512, truncation=True)
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )

    # Chat input
    query = st.text_input(" Ask a question about the PDF:")

    if query:
        with st.spinner("Thinking..."):
            response = qa.invoke({"query": query})
            st.markdown(f"** Answer:** {response['result']}")
