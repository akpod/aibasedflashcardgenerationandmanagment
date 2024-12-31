__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3

from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
import time


# Create directories if they don't exist
if not os.path.exists('files'):
    os.mkdir('files')

if not os.path.exists('db'):
    os.mkdir('db')

# Initialize vectorstore and LLM
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(persist_directory='db',
                                          embedding_function=OllamaEmbeddings(base_url='http://localhost:11434',
                                                                              model="llama3")
                                          )
if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(base_url="http://localhost:11434",
                                  model="llama3",
                                  verbose=True,
                                  callback_manager=CallbackManager(
                                      [StreamingStdOutCallbackHandler()]),
                                  )

# Initialize session state
if 'flashcards' not in st.session_state:
    st.session_state.flashcards = []

# Title of the app
st.title("Generate Flashcards from Your PDFs")

# Upload a PDF file
uploaded_file = st.file_uploader("Upload your PDF", type='pdf')

# Check if the uploaded file is already processed
if uploaded_file is not None:
    if not os.path.isfile("files/"+uploaded_file.name+".pdf"):
        with st.status("Analyzing your document..."):
            bytes_data = uploaded_file.read()
            f = open("files/"+uploaded_file.name+".pdf", "wb")
            f.write(bytes_data)
            f.close()
            loader = PyPDFLoader("files/"+uploaded_file.name+".pdf")
            data = loader.load()

            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len
            )
            all_splits = text_splitter.split_documents(data)

            # Create and persist the vector store
            st.session_state.vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=OllamaEmbeddings(model="llama3")
            )
            st.session_state.vectorstore.persist()

    # Create retriever from vectorstore
    st.session_state.retriever = st.session_state.vectorstore.as_retriever()

    # Initialize the QA chain (used for extracting flashcard info)
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
        )

    # Generate Flashcards (ask questions about the document)
    if st.button("Generate Flashcards"):
        st.session_state.flashcards.clear()  # Clear any previous flashcards
        with st.spinner("Generating flashcards..."):
            # Make sure 'all_splits' is defined here if the PDF is uploaded
            if 'all_splits' in locals() or 'all_splits' in globals():
                for chunk in all_splits:
                    response = st.session_state.qa_chain(chunk['text'])
                    question = f"What is {response['result'][:50]}?"
                    answer = response['result'][50:]  # This is just an example; customize as needed
                    flashcard = {'question': question, 'answer': answer}
                    st.session_state.flashcards.append(flashcard)

        # Display Flashcards
        st.subheader("Generated Flashcards")
        for i, flashcard in enumerate(st.session_state.flashcards, 1):
            st.write(f"**Flashcard {i}:**")
            st.write(f"**Question:** {flashcard['question']}")
            st.write(f"**Answer:** {flashcard['answer']}")

else:
    st.write("Please upload a PDF file.")
