import os
from llama_index.llms.azure_inference import AzureAICompletionsModel
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.embeddings.azure_inference import AzureAIEmbeddingsModel
import chromadb

AZURE_META_API = os.getenv("AZURE_META_API")
AZURE_META_ENDPOINT = os.getenv("AZURE_META_ENDPOINT")
AZURE_COHERE_API = os.getenv("AZURE_COHERE_API")
AZURE_COHERE_ENDPOINT = os.getenv("AZURE_COHERE_ENDPOINT")

# Initialize chromadb_client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def initialize_llm():
    """Initialize and return the Azure AI completions model."""
    return AzureAICompletionsModel(
        endpoint = AZURE_META_ENDPOINT,
        credential = AZURE_META_API,
    )

def initialize_embed_model():
    """Initialize and return the Azure AI Embedding model."""
    return AzureAIEmbeddingsModel(
        endpoint = AZURE_COHERE_ENDPOINT,
        credential = AZURE_COHERE_API,
    )

def connect_chromadb_create_index(documents):
    """
    Connects to chromaDB vector stores for persistent storage.
    Creates and returns a VectorStore index from the documents.
    """

    try:
        chroma_collection = chroma_client.get_or_create_collection("given_doc")
        Settings.embed_model = initialize_embed_model()
        # set up ChromaVectorStore and load in data
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        return index
    except Exception as e:
        raise Exception(f"Error connecting to ChromaDB or creating index: {e}") from e
    

def clear_chromadb_db():
    """Clear the chromadb database"""
    try:
        chroma_client.delete_collection("given_doc")
    except Exception as e:
        raise Exception(f"Error clearing ChromaDB collection: {e}") from e
    