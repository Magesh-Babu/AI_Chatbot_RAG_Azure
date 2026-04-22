import os
import sys

# Must be set before any backend module is imported — chat.py validates these at import time
os.environ.setdefault("AZURE_META_API", "test-meta-api-key")
os.environ.setdefault("AZURE_META_ENDPOINT", "https://test-meta-endpoint.azure.com")
os.environ.setdefault("AZURE_COHERE_API", "test-cohere-api-key")
os.environ.setdefault("AZURE_COHERE_ENDPOINT", "https://test-cohere-endpoint.azure.com")

# Add the backend directory to sys.path so test files can import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
