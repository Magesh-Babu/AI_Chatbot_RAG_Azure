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

_REQUIRED_ENV_VARS = {
    "AZURE_META_API": AZURE_META_API,
    "AZURE_META_ENDPOINT": AZURE_META_ENDPOINT,
    "AZURE_COHERE_API": AZURE_COHERE_API,
    "AZURE_COHERE_ENDPOINT": AZURE_COHERE_ENDPOINT,
}

missing = [name for name, value in _REQUIRED_ENV_VARS.items() if not value]
if missing:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing)}. "
        "Please set them in your .env file or container environment."
    )

# Initialize chromadb_client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def initialize_llm():
    """Initialize and return the Azure AI completions model."""
    return AzureAICompletionsModel(
        endpoint=AZURE_META_ENDPOINT,
        credential=AZURE_META_API,
    )

def initialize_embed_model():
    """Initialize and return the Azure AI Embedding model."""
    return AzureAIEmbeddingsModel(
        endpoint=AZURE_COHERE_ENDPOINT,
        credential=AZURE_COHERE_API,
    )

def connect_chromadb_create_index(documents):
    """
    Connects to chromaDB vector stores for persistent storage.
    Creates and returns a VectorStore index from the documents.
    """
    if not documents:
        raise ValueError("Cannot create index: document list is empty.")

    try:
        chroma_collection = chroma_client.get_or_create_collection("given_doc")
        Settings.embed_model = initialize_embed_model()
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        return index
    except ValueError:
        raise
    except chromadb.errors.ChromaError as e:
        raise RuntimeError(f"ChromaDB error while creating index: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error creating index: {e}") from e


def clear_chromadb_db():
    """Clear the chromadb database."""
    try:
        chroma_client.delete_collection("given_doc")
    except chromadb.errors.ChromaError as e:
        raise RuntimeError(f"ChromaDB error while clearing collection: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error clearing ChromaDB: {e}") from e
