from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import pipeline

# Step 1: Load the PDF
loader = PyPDFLoader("paper 2.pdf")
pages = loader.load()

# Step 2: Split the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(pages)

# Step 3: Create embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embedding)

# ✅ Step 4: Local pipeline using HuggingFace transformers
hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512, truncation=True)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Step 5: Retrieval-based QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Step 6: Chat loop
print("\n🤖 Ask something about the PDF (type 'exit' to quit):")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        print("👋 Exiting chatbot. Bye!")
        break
    response = qa_chain.invoke({"query": query})
    print(f"\nBot: {response['result']}\n")
