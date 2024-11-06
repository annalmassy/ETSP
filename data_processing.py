import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
from typing import List, Tuple
from pypdf import PdfReader
import re

import os
from chromadb import Client

# Set the persistence directory within the current project folder
persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
chroma_client = Client(Settings(persist_directory=persist_dir))

print(f"Chroma initialized with persistence at {persist_dir}")

# Check if the directory was created
if os.path.exists(persist_dir):
    print("Persistence directory created successfully!")
else:
    print("Failed to create persistence directory.")



def parse_pdf(file: BytesIO, filename: str) -> List[Tuple[str, int]]:
    pdf = PdfReader(file)
    output = []
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append((text, i + 1))
    return output

def text_to_docs(text_pages: List[Tuple[str, int]], filename: str) -> List[Document]:
    docs = []
    for text, page_number in text_pages:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            doc = Document(page_content=chunk, metadata={"filename": filename, "page": page_number, "chunk": i})
            docs.append(doc)
    return docs

def docs_to_chroma_index(docs: List[Document], openai_api_key: str) -> Chroma:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    index = Chroma.from_documents(docs, embeddings, client=chroma_client, collection_name="nlp_documents")
    return index

from langchain_community.vectorstores import Chroma

def get_chroma_index_for_pdf(files, filenames, openai_api_key, persist_directory):
    """Creates or updates a Chroma index with provided PDF documents."""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    try:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Chroma with error: {e}")
    
    documents = []
    flagged_files = []
    for file, filename in zip(files, filenames):
        # Parse and check if file is NLP-relevant
        text_pages = parse_pdf(BytesIO(file), filename)
        if is_nlp_relevant(text_pages):
            documents.extend(text_to_docs(text_pages, filename))
        else:
            flagged_files.append(filename)

    # Only add documents if there are new ones
    if documents:
        vectordb.add_documents(documents)
    return vectordb, flagged_files

def is_nlp_relevant(text_pages: List[Tuple[str, int]]) -> bool:
    """Checks if document contains NLP-relevant content."""
    nlp_keywords = {"tokenization", "linguistics", "language", "parsing", "syntax", "semantics"}
    for text, _ in text_pages:
        if any(keyword in text.lower() for keyword in nlp_keywords):
            return True
    return False