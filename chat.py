import os
import streamlit as st
from llama_index.llms.azure_inference import AzureAICompletionsModel
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

AZURE_META_API = os.getenv("AZURE_META_API")
AZURE_META_ENDPOINT = os.getenv("AZURE_META_ENDPOINT")

def initialize_llm():
    """Initialize and return the Azure AI completions model."""
    return AzureAICompletionsModel(
        endpoint = AZURE_META_ENDPOINT,
        credential = AZURE_META_API,
    )

def connect_chromadb_create_index(documents):
    """
    Connects to chromaDB vector stores for persistent storage.
    Creates and returns a VectorStore index from the documents.
    """
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("given_doc")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    # set up ChromaVectorStore and load in data
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
    return index

def display_chat():
    """Displays chat messages stored in session state."""
    if "messages" not in st.session_state or not st.session_state.messages:
        # Initialize default general chat messages
        st.session_state.messages = [{"role": "assistant", "content": "Hello, How can i help you?"}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            

def clear_chat_history():
    """Clears the chat history and resets the session state."""
    st.session_state.messages = []
    if "chat_engine" in st.session_state:
        del st.session_state.chat_engine
    if "uploaded_file_path" in st.session_state:
        try:
            os.remove(st.session_state.uploaded_file_path)
        except FileNotFoundError:
            pass
        del st.session_state.uploaded_file_path

