import streamlit as st

from utils.conversational_chain import handle_user_question
from utils.csv_processing import process_csv
from utils.pdf_processing import pdf_read, get_chunks
from utils.vectorstore import save_to_vectorstore
from utils.web_processing import fetch_web_data


def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="RAG Chatbot")
    st.header("RAG Chatbot with PDF, CSV, and Web Data Processing 🤖")

    # Entry point for user question
    user_question = st.text_input("Ask me anything...")

    if user_question:
        with st.spinner("Processing..."):
            handle_user_question(user_question)
    
    # Sidebar for managing PDF, CSV, and Web data
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )
        if st.button("Submit and process PDFs"):
            with st.spinner("Processing PDFs..."):
                raw_text = pdf_read(pdf_docs)
                save_to_vectorstore(get_chunks(raw_text))
                st.success("PDFs processed and saved to FAISS.")

        csv_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"]
        )
        if st.button("Submit and process CSV"):
            with st.spinner("Processing CSV..."):
                csv_texts = process_csv(csv_file)
                combined_csv_text = " ".join(csv_texts)
                save_to_vectorstore(get_chunks(combined_csv_text))
                st.success("CSV processed and saved to FAISS.")
        
        web_url = st.text_input("Enter a Web URL")
        if st.button("Submit and process Web data"):
            with st.spinner("Processing Web data..."):
                try:
                    web_text = fetch_web_data(web_url)
                    save_to_vectorstore(get_chunks(web_text))
                    st.success("Web data processed and saved to FAISS.")
                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
