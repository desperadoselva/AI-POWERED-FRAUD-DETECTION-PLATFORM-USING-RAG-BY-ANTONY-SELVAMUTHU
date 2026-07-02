"""
==============================================================
AI Powered Fraud Detection Platform
vector_store.py

Description
------------
Handles all FAISS Vector Store operations.


==============================================================
"""

import os

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = "/content/drive/MyDrive"
VECTOR_STORE_PATH = os.path.join(
    BASE_DIR,
    "vectorstore"
)

os.makedirs(
    VECTOR_STORE_PATH,
    exist_ok=True
)


# ==========================================================
# Embedding Model
# ==========================================================

EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================================
# Vector Store Class
# ==========================================================

class VectorStoreManager:

    def __init__(self):

        self.embedding = EMBEDDING_MODEL

        self.vectorstore = None


    # ------------------------------------------------------
    # Create Vector Store
    # ------------------------------------------------------

    def create(self, documents):

        print("\nCreating FAISS Vector Store...")

        self.vectorstore = FAISS.from_documents(
            documents,
            self.embedding
        )

        print("Vector Store Created.")

        return self.vectorstore


    # ------------------------------------------------------
    # Save Vector Store
    # ------------------------------------------------------

    def save(self):

        if self.vectorstore is None:
            raise ValueError("Vector Store has not been created.")

        self.vectorstore.save_local(
            VECTOR_STORE_PATH
        )

        print("Vector Store Saved.")

        print(VECTOR_STORE_PATH)


    # ------------------------------------------------------
    # Load Vector Store
    # ------------------------------------------------------

    def load(self):

        print("Loading Vector Store...")

        self.vectorstore = FAISS.load_local(

            VECTOR_STORE_PATH,

            self.embedding,

            allow_dangerous_deserialization=True

        )

        print("Vector Store Loaded.")

        return self.vectorstore


    # ------------------------------------------------------
    # Add New Documents
    # ------------------------------------------------------

    def add_documents(self, documents):

        if self.vectorstore is None:
            self.load()

        self.vectorstore.add_documents(
            documents
        )

        self.save()

        print("Documents Added Successfully.")


    # ------------------------------------------------------
    # Similarity Search
    # ------------------------------------------------------

    def similarity_search(

        self,

        query,

        k=4

    ):

        if self.vectorstore is None:
            self.load()

        return self.vectorstore.similarity_search(
            query,
            k=k
        )


    # ------------------------------------------------------
    # Similarity Search with Score
    # ------------------------------------------------------

    def similarity_search_with_score(

        self,

        query,

        k=4

    ):

        if self.vectorstore is None:
            self.load()

        return self.vectorstore.similarity_search_with_score(
            query,
            k=k
        )


    # ------------------------------------------------------
    # LangChain Retriever
    # ------------------------------------------------------

    def as_retriever(

        self,

        k=4

    ):

        if self.vectorstore is None:
            self.load()

        return self.vectorstore.as_retriever(

            search_kwargs={

                "k": k

            }

        )


    # ------------------------------------------------------
    # Retrieve Context
    # ------------------------------------------------------

    def get_context(

        self,

        query,

        k=4

    ):

        docs = self.similarity_search(
            query,
            k
        )

        context = ""

        for doc in docs:

            context += doc.page_content

            context += "\n\n"

        return context


# ==========================================================
# Singleton
# ==========================================================

vector_manager = VectorStoreManager()


# ==========================================================
# Helper Functions
# ==========================================================

def create_vectorstore(documents):

    vector_manager.create(documents)

    vector_manager.save()


def load_vectorstore():

    return vector_manager.load()


def retrieve_context(

    query,

    k=4

):

    return vector_manager.get_context(

        query,

        k

    )


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    vector_manager.load()

    question = "Explain RBI guidelines for high-value transactions."

    print("=" * 60)

    print(question)

    print("=" * 60)

    context = vector_manager.get_context(

        question,

        k=3

    )

    print(context)