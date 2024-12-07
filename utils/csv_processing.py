import os
import pandas as pd

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))


def process_csv(csv_file):
    """
    Read the text data from the CSV file.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)
    # Convert the DataFrame rows to text
    rows = df.apply(lambda row: row.to_dict(), axis=1)
    texts = [str(row) for row in rows]
    return texts
