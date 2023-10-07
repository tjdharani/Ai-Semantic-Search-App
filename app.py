import os
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.title("AI Semantic Search App 🔍")

st.sidebar.title("Articles URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

submitted_url = st.sidebar.button("Submit")

file_path = "data.pkl"
main_placeholder = st.empty()
llm = OpenAI(temperature=0.7, max_tokens=500)

if submitted_url:
    # data loading
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...⏳")
    data = loader.load()

    # data splitting 
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("Text Splitting...⏳")
    docs = text_splitter.split_documents(data)

    # data embeddings and saving as FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Vector Embedding...⏳")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)


query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"])

            # Give Source
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
