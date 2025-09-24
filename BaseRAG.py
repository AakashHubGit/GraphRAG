# base_rag.py - Fixed version

import os
import asyncio
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from rouge_score import rouge_scorer
# Use the updated import for HuggingFaceEmbeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback for older versions
    from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_text(text)

def create_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_rag_chain():
    template = """
    You are an AI assistant. Use the provided context from the document to answer the question.
    If the answer is not in the context, say "I couldn't find that in the document."

    Context:
    {context}

    Question: {question}

    Answer:
    """
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt | llm

def ask_question(question, k=3):
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(question, k=k)
    context = " ".join([doc.page_content for doc in docs])

    chain = get_rag_chain()
    answer = chain.invoke({"context": context, "question": question}).content
    return answer