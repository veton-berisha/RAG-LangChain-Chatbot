import os
import streamlit as st

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate

from utils.vectorstore import load_vectorstore

load_dotenv()


def get_conversational_chain(tool, question):
    """
    Configure and run the conversational chain.
    """
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        verbose=True
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant. Answer the question as detailed as possible from the provided context. 
                If the answer is not in the context, say: "Answer is not available in the context"."""
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, [tool], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=True)
    response = agent_executor.invoke({"input": question})
    st.write("Reply: ", response["output"])


def handle_user_question(user_question):
    """
    Handle the user question and get the response.
    """
    vectorstore = load_vectorstore()
    if vectorstore is None:
        st.error("Vector store is empty. Please upload PDF, CSV, or Web data to proceed.")
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "multi_source_extractor",
        "This tool is to give answers to queries from the PDF, CSV, and Web data."
    )
    get_conversational_chain(retriever_tool, user_question)
