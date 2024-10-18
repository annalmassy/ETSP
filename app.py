import streamlit as st
from openai import OpenAI
import os
from data_processing import get_index_for_pdf
import torch
import torch.nn.functional as F
from rouge import Rouge
import pandas as pd
import plotly.express as px

# Set up OpenAI client
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
# Set the title for the Streamlit app
st.title("Educational NLP Learning Chatbot")

# Cached function to create a vector database for the provided PDF files
@st.cache_resource
def create_educational_vectordb(file, filename):
    # Show a spinner while creating the vectordb
    with st.spinner(f"Creating vector database for {filename}..."):
        try:
            vectordb = get_index_for_pdf([file], [filename], client.api_key)
            return vectordb
        except Exception as e:
            st.error(f"Error creating vector database for {filename}: {e}")
            return None

# Upload PDF files using Streamlit's file uploader
st.subheader("Upload NLP Learning Materials")
pdf_files = st.file_uploader("", type="pdf", accept_multiple_files=True)

# Create vector databases for all uploaded files
vectordbs = {}
if pdf_files:
    vectordbs.clear()
    for file in pdf_files:
        vectordb = create_educational_vectordb(file.getvalue(), file.name)
        if vectordb is not None:
            vectordbs[file.name] = vectordb

# Define the template for the chatbot prompt
prompt_template = """
    You are an expert in Natural Language Processing (NLP). Your task is to assist students by providing explanations and insights on various NLP topics based on the provided study materials. Provide clear and concise answers, and reference the relevant section or page from the uploaded materials when needed.
"""

# Get the current prompt from the session state or set a default value
if "prompt" not in st.session_state:
    st.session_state["prompt"] = [{"role": "system", "content": "none"}]

if "result_rag" not in st.session_state:
    st.session_state["result_rag"] = ""

if "result_gpt3" not in st.session_state:
    st.session_state["result_gpt3"] = ""

prompt = st.session_state["prompt"]

# Display previous chat messages
for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Get the user's question using Streamlit's chat input
question = st.chat_input("Ask anything")

# Checkboxes to select which models to use
use_rag = st.checkbox("Use RAG (GPT)")
use_gpt3 = st.checkbox("Use GPT-3.5")
use_expert = st.checkbox("Use Expert Answer")

# Select the document to use if any model is selected
if use_rag or use_gpt3:
    doc_choice = st.selectbox("Used document", list(vectordbs.keys()))
    vectordb = vectordbs.get(doc_choice, None)
    if not vectordb:
        st.error("You need to provide a PDF")
        st.stop()

# Handle the user's question
if question:
    # Reset the session state for new questions
    st.session_state["result_rag"] = ""
    st.session_state["result_gpt3"] = ""
    
    # Display the user's question
    with st.chat_message("user"):
        st.write(question)
    prompt.append({"role": "user", "content": question})

    # Prepare the prompts for both models
    prompt_rag = prompt.copy()
    prompt_gpt3 = prompt.copy()

    if use_rag:
        # Search the vectordb for similar content to the user's question
        search_results = vectordb.similarity_search(question, k=3)
        pdf_extract = "\n".join([result.page_content for result in search_results])

        # Update the prompt with the pdf extract
        prompt_rag[0] = {
            "role": "system",
            "content": prompt_template.format(pdf_extract=pdf_extract),
        }

    if use_gpt3:
        # Use the same prompt for GPT-3.5 but without the PDF extract
        prompt_gpt3[0] = {
            "role": "system",
            "content": "You are a Credit Analyst who evaluates financial information and risks. Provide concise and focused answers, emphasizing financial implications, risk assessment, and relevant financial metrics.",
        }

    # Generate the RAG answer
    if use_rag:
        with st.chat_message("assistant"):
            botmsg_rag = st.empty()
            botmsg_rag.write("RAG Answer: thinking...")

        response_rag = []
        result_rag = ""
        for chunk in client.chat.completions.create(
            model="gpt-4o", messages=prompt_rag, stream=True
        ):
            text = chunk.choices[0].delta.content
            if text is not None:
                response_rag.append(text)
                result_rag = "".join(response_rag).strip()

        # Update the message with the final response
        botmsg_rag.write(f"RAG Answer:\n{result_rag}")

        # Store the RAG result
        st.session_state["result_rag"] = result_rag

    # Generate the GPT-3.5 answer
    if use_gpt3:
        with st.chat_message("assistant"):
            botmsg_gpt3 = st.empty()
            botmsg_gpt3.write("GPT-3.5 Answer: thinking...")

        response_gpt3 = []
        result_gpt3 = ""
        for chunk in client.chat.completions.create(
            model="gpt-3.5-turbo", messages=prompt_gpt3, stream=True
        ):
            text = chunk.choices[0].delta.content
            if text is not None:
                response_gpt3.append(text)
                result_gpt3 = "".join(response_gpt3).strip()

        # Update the message with the final response
        botmsg_gpt3.write(f"GPT-3.5 Answer:\n{result_gpt3}")

        # Store the GPT-3.5 result
        st.session_state["result_gpt3"] = result_gpt3

# Display the final results if they are present
if st.session_state["result_rag"]:
    st.subheader("RAG Answer:")
    st.write(st.session_state["result_rag"])

if st.session_state["result_gpt3"]:
    st.subheader("GPT-3.5 Answer:")
    st.write(st.session_state["result_gpt3"])

# Expert Answer Input
if use_expert:
    expert_answer = st.text_area("Enter Expert Answer:")
    if st.button("Evaluate Expert Answer"):
        if expert_answer and st.session_state["result_rag"]:
            try:
                rouge = Rouge()
                scores_expert = rouge.get_scores(expert_answer, st.session_state["result_rag"], avg=True)
                st.write("ROUGE Scores (Expert vs RAG):")
                st.write(scores_expert)

                # Visualization
                df = pd.DataFrame(scores_expert).T.reset_index().melt(id_vars="index")
                df.columns = ["metric", "score_type", "value"]
                fig = px.bar(df, x="metric", y="value", color="score_type", barmode="group",
                             title="ROUGE Scores for Expert vs RAG Answer")
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error during evaluation: {e}")
        else:
            st.write("Expert answer and RAG answer are required for evaluation.")

# Simple evaluation
if use_rag and use_gpt3:
    if st.button("Evaluate Responses"):
        result_rag = st.session_state.get("result_rag", "")
        result_gpt3 = st.session_state.get("result_gpt3", "")

        if result_rag and result_gpt3:
            try:
                rouge = Rouge()
                scores = rouge.get_scores(result_rag, result_gpt3, avg=True)
                st.write("ROUGE Scores (RAG vs GPT-3.5):")
                st.write(scores)

                # Visualization
                df = pd.DataFrame(scores).T.reset_index().melt(id_vars="index")
                df.columns = ["metric", "score_type", "value"]
                fig = px.bar(df, x="metric", y="value", color="score_type", barmode="group",
                             title="ROUGE Scores for RAG vs GPT-3.5 Answer")
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error during evaluation: {e}")
        else:
            st.write("Both responses are required for evaluation.")
