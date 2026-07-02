!pip install pypdf
"""
=========================================================
AI Powered Fraud Detection Platform
ingest.py

Description
------------
Loads PDF documents from data/knowledge,
splits them into chunks,
creates embeddings,
and stores them in FAISS.

Author : Antony Selvamuthu
=========================================================
"""

import os

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings



# ==========================================================
# Project Paths
# ==========================================================

# Corrected BASE_DIR to point to /content/drive/MyDrive
BASE_DIR = "/content/drive/MyDrive"

KNOWLEDGE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "knowledge"
)

VECTOR_DB = os.path.join(
    BASE_DIR,
    "vectorstore"
)



# ==========================================================
# Embedding Model
# ==========================================================

embedding_model = HuggingFaceEmbeddings(

    model_name="sentence-transformers/all-MiniLM-L6-v2"

)



# ==========================================================
# Load PDFs
# ==========================================================

def load_documents():

    documents = []

    # Ensure the KNOWLEDGE_DIR exists before listing its contents
    if not os.path.exists(KNOWLEDGE_DIR):
        print(f"Warning: KNOWLEDGE_DIR '{KNOWLEDGE_DIR}' does not exist. No PDF files will be loaded.")
        return []

    pdf_files = [

        file

        for file in os.listdir(KNOWLEDGE_DIR)

        if file.endswith(".pdf")

    ]

    print(f"\nFound {len(pdf_files)} PDF files.\n")

    for pdf in pdf_files:

        path = os.path.join(

            KNOWLEDGE_DIR,

            pdf

        )

        print(f"Loading {pdf}")

        loader = PyPDFLoader(path)

        docs = loader.load()

        documents.extend(docs)

    return documents



# ==========================================================
# Split Documents
# ==========================================================

def split_documents(documents):

    if not documents:
        print("No documents to split.")
        return []

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200

    )

    chunks = splitter.split_documents(documents)

    print(f"\nCreated {len(chunks)} chunks.\n")

    return chunks



# ==========================================================
# Create FAISS Index
# ==========================================================

def create_vector_store(chunks):

    if not chunks:
        print("No chunks to create vector store from. Skipping FAISS creation.")
        return

    print("Creating FAISS Vector Store...")

    vectorstore = FAISS.from_documents(

        chunks,

        embedding_model

    )

    os.makedirs(VECTOR_DB, exist_ok=True)

    vectorstore.save_local(VECTOR_DB)

    print("\nVector Database Saved.")

    print(VECTOR_DB)



# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)

    print("Fraud Knowledge Base Ingestion")

    print("=" * 60)

    docs = load_documents()

    print(f"Loaded {len(docs)} pages.")

    chunks = split_documents(docs)

    create_vector_store(chunks)

    print("\nKnowledge Base Ready!")



if __name__ == "__main__":

    main()