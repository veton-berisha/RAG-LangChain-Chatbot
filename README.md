# RAG Chatbot Project with LangChain, OpenAI, FAISS and Streamlit

A chatbot based on Retrieval-Augmented Generation (RAG) that uses LangChain, OpenAI, FAISS and Streamlit to answer questions based on custom data (PDF, CSV, Web Search).

## Installation

1. **Create a virtual environment :**
```bash
python -m venv .venv
```

2. **Activate the virtual environment :**
```bash
source .venv/bin/activate # Linux
.venv\Scripts\activate # Windows
```

3. **Install the dependencies :**
```bash
pip install -r requirements.txt
```

4. **Add the OpenAI API key :**
Create a `.env` file at the root of the project and add the OpenAI API key :
```bash
OPENAI_API_KEY=yoursecretkey
```

5. **Run the Streamlit app :**
```bash
streamlit run app.py
```

## Usage

1. **Import data :**
Import data from a PDF, CSV or Web Search to train the chatbot.

2. **Ask questions :**
Ask questions to the chatbot and get answers based on the imported data.