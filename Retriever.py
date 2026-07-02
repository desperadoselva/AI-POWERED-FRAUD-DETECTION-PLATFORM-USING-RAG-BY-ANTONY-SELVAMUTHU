"""
==============================================================
AI Powered Fraud Detection Platform
retriever.py

Description
------------
Loads the FAISS Vector Database and retrieves
relevant banking knowledge for RAG.

Author : Antony Selvamuthu
==============================================================
"""

import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = "/content/drive/MyDrive"

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
# Retriever Class
# ==========================================================

class FraudRetriever:

    def __init__(self):

        print("Loading FAISS Vector Store...")

        self.vectorstore = FAISS.load_local(

            VECTOR_DB,

            embedding_model,

            allow_dangerous_deserialization=True

        )

        print("Vector Store Loaded Successfully.\n")


    # ======================================================
    # Similarity Search
    # ======================================================

    def search(

        self,

        query,

        k=4

    ):

        docs = self.vectorstore.similarity_search(

            query,

            k=k

        )

        return docs


    # ======================================================
    # Similarity Search with Score
    # ======================================================

    def search_with_score(

        self,

        query,

        k=4

    ):

        docs = self.vectorstore.similarity_search_with_score(

            query,

            k=k

        )

        return docs


    # ======================================================
    # Retriever for LangChain
    # ======================================================

    def get_retriever(

        self,

        k=4

    ):

        return self.vectorstore.as_retriever(

            search_kwargs={

                "k": k

            }

        )


    # ======================================================
    # Retrieve Context
    # ======================================================

    def get_context(

        self,

        query,

        k=4

    ):

        docs = self.search(

            query,

            k

        )

        context = ""

        for doc in docs:

            context += doc.page_content

            context += "\n\n"

        return context


# ==========================================================
# Helper Functions
# ==========================================================

retriever = FraudRetriever()


def retrieve(query):

    return retriever.get_context(query)


def retrieve_documents(query):

    return retriever.search(query)


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    question = """

    Why is a repeated login to my bank account
    at 2 AM considered suspicious?

    """

    print("=" * 60)

    print("Question")

    print("=" * 60)

    print(question)

    print()

    docs = retriever.search(question)

    print("=" * 60)

    print("Retrieved Documents")

    print("=" * 60)

    for i, doc in enumerate(docs, 1):

        print(f"\nDocument {i}")

        print("-" * 40)

        print(doc.page_content[:800])

        print()