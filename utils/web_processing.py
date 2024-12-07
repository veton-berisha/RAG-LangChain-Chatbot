import os
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))


def fetch_web_data(url):
    """
    Fetches the text data from a given URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Verify if the request was successful
        soup = BeautifulSoup(response.text, "html.parser")
        # Extract text from the HTML content
        text = soup.get_text(separator="\n")
        return text.strip()
    except Exception as e:
        raise Exception(f"Erreur lors de la récupération de la page web : {e}")
