import os
from llama_index.llms.azure_inference import AzureAICompletionsModel
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.embeddings.azure_inference import AzureAIEmbeddingsModel
import chromadb
from logging_config import get_logger

logger = get_logger(__name__)

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
    logger.error("Missing required environment variables: %s", ", ".join(missing))
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing)}. "
        "Please set them in your .env file or container environment."
    )

logger.info("Azure environment variables validated successfully.")

# Initialize chromadb_client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
logger.info("ChromaDB persistent client initialised at ./chroma_db")


def initialize_llm():
    """Initialize and return the Azure AI completions model."""
    logger.info("Initialising Azure AI completions model (Llama 3).")
    return AzureAICompletionsModel(
        endpoint=AZURE_META_ENDPOINT,
        credential=AZURE_META_API,
    )


def initialize_embed_model():
    """Initialize and return the Azure AI Embedding model."""
    logger.info("Initialising Azure AI embeddings model (Cohere).")
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

    logger.info("Creating ChromaDB vector index from %d document(s).", len(documents))
    try:
        chroma_collection = chroma_client.get_or_create_collection("given_doc")
        Settings.embed_model = initialize_embed_model()
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        logger.info("Vector index created successfully.")
        return index
    except ValueError:
        raise
    except chromadb.errors.ChromaError as e:
        logger.error("ChromaDB error while creating index: %s", e, exc_info=True)
        raise RuntimeError(f"ChromaDB error while creating index: {e}") from e
    except Exception as e:
        logger.error("Unexpected error creating index: %s", e, exc_info=True)
        raise RuntimeError(f"Unexpected error creating index: {e}") from e


def clear_chromadb_db():
    """Clear the chromadb database."""
    logger.info("Clearing ChromaDB collection 'given_doc'.")
    try:
        chroma_client.delete_collection("given_doc")
        logger.info("ChromaDB collection cleared successfully.")
    except chromadb.errors.ChromaError as e:
        logger.error("ChromaDB error while clearing collection: %s", e, exc_info=True)
        raise RuntimeError(f"ChromaDB error while clearing collection: {e}") from e
    except Exception as e:
        logger.error("Unexpected error clearing ChromaDB: %s", e, exc_info=True)
        raise RuntimeError(f"Unexpected error clearing ChromaDB: {e}") from e
