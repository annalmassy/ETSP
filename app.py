import subprocess
import sys

# List of required packages
required_packages = [
    "chromadb",
    "openai",
    "streamlit",
    "langchain",
    "langchain_openai",  # New package import for updated OpenAIEmbeddings
    "pandas",
    "plotly",
    "torch",
]

# Install missing packages
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


import streamlit as st
import os
from openai import OpenAI
import openai
from data_processing import get_chroma_index_for_pdf  # Updated for Chroma usage
import torch
import torch.nn.functional as F
from rouge import Rouge
import pandas as pd
import plotly.express as px
from test import  conduct_tests

# Set up OpenAI client with API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set the title for the Streamlit app
st.title("Educational NLP Learning Chatbot")

# Define persistence directory for Chroma database
persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")

# Hardcoded PDF for initial vector database
hardcoded_pdf = "An_Introduction_to_Language_and_Linguistics.pdf"
with open(hardcoded_pdf, "rb") as f:
    hardcoded_pdf_data = f.read()
hardcoded_pdf_name = "An Introduction to Language and Linguistics.pdf"

# Cached function to create Chroma vector database
@st.cache_resource
def create_educational_vectordb(files, filenames):
    with st.spinner("Creating vector database for all documents..."):
        try:
            vectordb, flagged_files = get_chroma_index_for_pdf(files, filenames, openai.api_key, persist_directory)
            if not vectordb:
                st.error("Failed to create vector database.")
            return vectordb, flagged_files
        except Exception as e:
            st.error(f"Error creating vector database: {e}")
            return None, []

# Initialize vector database with the hardcoded document
files = [hardcoded_pdf_data]
filenames = [hardcoded_pdf_name]

# UI for additional PDF uploads
st.subheader("Upload Additional NLP Learning Materials")
pdf_files = st.file_uploader("", type="pdf", accept_multiple_files=True)

# Add user-uploaded documents to the list
if pdf_files:
    for file in pdf_files:
        files.append(file.getvalue())
        filenames.append(file.name)

# Create or update the vector database with all documents
vectordb, flagged_files = create_educational_vectordb(files, filenames)

# Display any flagged files (non-NLP relevant)
if flagged_files:
    st.warning("The following files were flagged as non-NLP relevant and were not added to the database:")
    for file in flagged_files:
        st.write(file)


# Display RAG enable/disable option
use_rag = st.checkbox("Enable RAG (GPT with Retrieval)")
st.write("Chatbot based on:", "Hardcoded PDF and uploaded materials" if use_rag else "Standard GPT-4o Mini")

# Define prompt template for RAG usage
prompt_template = "You are an NLP expert. Answer questions clearly and concisely, referencing the uploaded materials when RAG is enabled."

# Initialize chat history in session state if not already present
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Run Test
#conduct_tests(vectordb, use_rag, prompt_template, client)

# Chat input for the user’s question
user_input = st.chat_input("Ask anything about NLP:")

if user_input:
    # Add user's message to the chat history
    st.session_state["chat_history"].append({"role": "user", "content": user_input})

    if use_rag and vectordb:
        # Using RAG to search in Chroma database
        search_results = vectordb.similarity_search(user_input, k=3)
        
        # Constructing RAG context with source references
        pdf_extract = ""
        for result in search_results:
            page_content = result.page_content
            filename = result.metadata.get("filename", "unknown document")
            page = result.metadata.get("page", "unknown page")
            pdf_extract += f"{page_content} [Source: {filename}, Page: {page}]\n\n"
        
        # Prompt for RAG model
        prompt_with_rag = [
            {"role": "system", "content": prompt_template},
            {"role": "assistant", "content": pdf_extract},
            {"role": "user", "content": user_input}
        ]
        
        # Generate RAG response
        response_rag = []
        for chunk in client.chat.completions.create(
            model="gpt-4o", messages=prompt_with_rag, stream=True
        ):
            text = chunk.choices[0].delta.content
            if text:
                response_rag.append(text)
        result_rag = "".join(response_rag).strip()
        st.session_state["chat_history"].append({"role": "assistant", "content": result_rag})
    else:
        # Standard GPT-4o Mini without RAG
        prompt_basic = [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": user_input}
        ]
        response_gpt3 = []
        for chunk in client.chat.completions.create(
            model="gpt-4o", messages=prompt_basic, stream=True
        ):
            text = chunk.choices[0].delta.content
            if text:
                response_gpt3.append(text)
        result_gpt3 = "".join(response_gpt3).strip()
        st.session_state["chat_history"].append({"role": "assistant", "content": result_gpt3})

# Display the entire chat history
for message in st.session_state["chat_history"]:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])

# Expert Answer Input and ROUGE Analysis (commented out to hide on UI)
# if False:  
#     expert_answer = st.text_area("Enter Expert Answer:")
#     if st.button("Evaluate Expert Answer"):
#         if expert_answer and result_rag:
#             try:
#                 rouge = Rouge()
#                 scores_expert = rouge.get_scores(expert_answer, result_rag, avg=True)
#                 st.write("ROUGE Scores (Expert vs RAG):", scores_expert)
#                 # Visualization code here
#             except Exception as e:
#                 st.error(f"Error during evaluation: {e}")
