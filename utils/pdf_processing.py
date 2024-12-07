from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


def pdf_read(pdf_docs):
    """
    Read the text from the PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Get the text chunks from the input text.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)
