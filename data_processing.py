import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
from typing import List, Tuple
import re
import os
import streamlit as st
from pypdf import PdfReader

# Set the persistence directory within the current project folder
persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
chroma_client = chromadb.Client(Settings(
    persist_directory=persist_dir,
    is_persistent=True
))

def parse_pdf(file: BytesIO, filename: str) -> List[Tuple[str, int]]:
    """
    Parse PDF file and extract text with proper cleanup.
    
    Args:
        file: BytesIO object containing PDF data
        filename: Name of the PDF file
        
    Returns:
        List of tuples containing (text, page_number)
    """
    pdf = PdfReader(file)
    output = []
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        # Clean up hyphenation and newlines
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append((text, i + 1))
    return output

def text_to_docs(text_pages: List[Tuple[str, int]], filename: str) -> List[Document]:
    """
    Convert text pages to LangChain documents with metadata.
    
    Args:
        text_pages: List of tuples containing (text, page_number)
        filename: Name of the source file
        
    Returns:
        List of Document objects
    """
    docs = []
    for text, page_number in text_pages:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "filename": filename,
                    "page": page_number,
                    "chunk": i
                }
            )
            docs.append(doc)
    return docs

@st.cache_resource
def create_educational_vectordb(files, filenames):
    """
    Create or update vector database with uploaded files.
    
    Args:
        files: List of file contents
        filenames: List of file names
        
    Returns:
        Tuple of (vectordb, flagged_files)
    """
    with st.spinner("Creating vector database for all documents..."):
        try:
            vectordb, flagged_files = get_chroma_index_for_pdf(files, filenames, os.getenv("OPENAI_API_KEY"), persist_dir)
            if not vectordb:
                st.error("Failed to create vector database.")
            return vectordb, flagged_files
        except Exception as e:
            st.error(f"Error creating vector database: {e}")
            return None, []

def get_chroma_index_for_pdf(files, filenames, openai_api_key: str, persist_directory: str):
    """
    Creates or updates Chroma index with provided PDF documents.
    
    Args:
        files: List of file contents
        filenames: List of file names
        openai_api_key: OpenAI API key
        persist_directory: Directory for persisting the vector database
        
    Returns:
        Tuple of (vectordb, flagged_files)
    """
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    try:
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client=chroma_client,
            collection_name="nlp_documents"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Chroma with error: {e}")
    
    documents = []
    flagged_files = []
    
    for file, filename in zip(files, filenames):
        try:
            # Parse and check if file is NLP-relevant
            text_pages = parse_pdf(BytesIO(file), filename)
            if is_nlp_relevant(text_pages):
                documents.extend(text_to_docs(text_pages, filename))
            else:
                flagged_files.append(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            flagged_files.append(filename)

    # Only add documents if there are new ones
    if documents:
        vectordb.add_documents(documents)
        
    return vectordb, flagged_files

def is_nlp_relevant(text_pages: List[Tuple[str, int]]) -> bool:
    """
    Check if document contains NLP-relevant content.
    
    Args:
        text_pages: List of tuples containing (text, page_number)
        
    Returns:
        Boolean indicating if content is NLP-relevant
    """
    nlp_keywords = {
        "tokenization", "linguistics", "language", "parsing", "syntax", 
        "semantics", "nlp", "natural language processing", "computational linguistics",
        "text analysis", "language model", "machine translation", 
        "sentiment analysis", "named entity recognition", "part of speech",
        "morphology", "phonetics", "phonology", "pragmatics", "corpus"
    }
    
    for text, _ in text_pages:
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in nlp_keywords):
            return True
    return False