import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))


def save_to_vectorstore(text_chunks):
    """
    Save the text chunks to the FAISS vector store.
    """
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_db")


def load_vectorstore():
    """
    Load the FAISS vector store.
    """
    return FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
