import json
import boto3
import os
import sys
import streamlit as st

# Use Amazon Titan for generating embeddings

from langchain_community.embeddings import BedrockEmbeddings
# from langchain_community.llms.bedrock import Bedrock
from langchain_aws import BedrockLLM, ChatBedrock
from langchain_openai import ChatOpenAI


# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# convert to vector embeddings and store them.
from langchain_community.vectorstores import FAISS
# will try with mongodb too.


# LLMs and Chaining
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, RunnableSequence
from langchain_core.output_parsers import StrOutputParser


# Bedrock client

bedrock = boto3.client(service_name= "bedrock-runtime", region_name= "us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


# Data Ingestion
def data_ingestion():
    # Load the PDF files from the directory.
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    # Split the documents into smaller chunks.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1200, chunk_overlap= 200)
    docs = text_splitter.split_documents(documents)
    
    return docs

## Vector Embeddings and Vector Store

def get_vector_store(docs):
    vector_store = FAISS.from_documents(
        documents=docs,
        embedding=bedrock_embeddings
    )
    vector_store.save_local("faiss_index")
    
def get_llama_llm():
    llm = BedrockLLM(
        model_id= "us.meta.llama3-1-8b-instruct-v1:0", 
        client= bedrock,
        model_kwargs= {"max_gen_len": 2048},
        streaming= True
    )

    return llm

def get_openai_model():
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        max_tokens=2000,
    )
    
    return llm

# create the prompt template
messages = [
    # ("system", "You're a helpful assistant that answers questions from the context in the provided documents. Make sure to answer the questions accurately. You're allowed to use external knowledge too with the knowledge from the documents."),
    ("human", """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
                Question: {question} 
                Context: {context} 
                Answer:""")
]

prompt = ChatPromptTemplate.from_messages(messages)
    
    
def get_response_llm(llm, vectorstore_faiss, query):
    retriever = vectorstore_faiss.as_retriever(
        search_type= "similarity",
        search_kwargs= {"k": 3},
    )
    
    qa_retriever = {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    
    # create a chain to get the response
    response_chain = qa_retriever | prompt | llm | StrOutputParser()
    
    return response_chain.invoke(query)


def main():
    st.set_page_config("Chat with Research Papers")
    st.header("Chat with Research Papers using ChatGPT-4o or AWS Bedrock")
    
    user_question = st.text_input("Ask a question about the research papers", placeholder="Type your question here...")
    
    with st.sidebar:
        st.title("Update or create a vector store: ")
        
        if st.button("Update vector store"):
            st.spinner("Updating the vector store...")
            docs = data_ingestion()
            get_vector_store(docs)
            st.write("Vector store updated successfully!")

    if st.button("Llama Output"):
        with st.spinner("Generating response..."):
            vectorstore_faiss = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama_llm()
            
            # faiss_index = get_vector_store(docs)
            response = get_response_llm(llm, vectorstore_faiss, user_question)
            st.write(response)
            st.success("Response generated successfully!")
            
    if st.button("OpenAI Output"):
        with st.spinner("Generating response..."):
            vectorstore_faiss = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_openai_model()
            
            # faiss_index = get_vector_store(docs)
            response = get_response_llm(llm, vectorstore_faiss, user_question)
            st.write(response)
            st.success("Response generated successfully!")
        
if __name__ == "__main__":
    main()